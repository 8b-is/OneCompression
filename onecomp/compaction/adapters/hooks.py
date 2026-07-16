"""ARC PostToolUse hook adapters for Claude Code and Codex.

The command entry point reads one hook event from stdin, stores a long tool
response in a per-session WAL, and emits the platform-specific JSON that
replaces the model-visible result with a bounded citation.  A paired recall
entry point prints the exact stored response on demand.

These adapters operate at tool-result ingestion time.  They do not pretend
that merely exposing the OneComp MCP server can rewrite a host application's
already-owned conversation history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Literal, Optional

from ..engine import ArcEngine, ArcEngineConfig
from ..protocol import format_citation
from ..store import ObsStore

_OBS_ID_RE = re.compile(r"^[0-9a-fA-F]{4,40}$")
_RECALL_COMMANDS = ("onecomp-arc-recall", "onecomp.compaction.adapters.hooks recall")


def process_post_tool_use(
    payload: dict[str, Any],
    *,
    platform: Literal["claude", "codex"],
    result_max_chars: int = 500,
    hook_root: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Store and replace one platform PostToolUse response.

    Returns an empty object when the response is short, unsupported, or is the
    paired recall command itself.  Hook callers can therefore configure a broad
    matcher while small ordinary results pass through unchanged.
    """
    if platform not in {"claude", "codex"}:
        raise ValueError("platform must be 'claude' or 'codex'")
    if not isinstance(payload, dict):
        raise ValueError("hook payload must be an object")
    if not isinstance(result_max_chars, int) or isinstance(result_max_chars, bool):
        raise ValueError("result_max_chars must be a non-negative integer")
    if result_max_chars < 0:
        raise ValueError("result_max_chars must be a non-negative integer")

    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("hook payload requires a non-empty session_id")
    tool_name = payload.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("hook payload requires a non-empty tool_name")
    tool_input = payload.get("tool_input", {})
    command = _command_signature(tool_name, tool_input)
    if any(marker in command for marker in _RECALL_COMMANDS):
        return {}

    response = payload.get("tool_response")
    content = _lossless_json_text(response)
    if len(content) <= result_max_chars:
        return {}

    wal_path = _wal_path(session_id, hook_root)
    wal_path.touch(mode=0o600, exist_ok=True)
    try:
        wal_path.chmod(0o600)
    except OSError:
        pass
    engine = ArcEngine.from_wal(
        str(wal_path),
        ArcEngineConfig(
            result_max_chars=result_max_chars,
            wal_path=str(wal_path),
        ),
    )
    try:
        record = engine.record_observation(
            command=command,
            content=content,
            returncode=_returncode(response),
            extra={
                "platform": platform,
                "tool_name": tool_name,
                "tool_use_id": str(payload.get("tool_use_id", "")),
            },
        )
    finally:
        engine.close()
    try:
        wal_path.chmod(0o600)
    except OSError:
        pass

    citation = format_citation(
        obs_id=record.obs_id,
        cmd_signature=record.cmd_signature,
        returncode=record.returncode,
        line_count=record.line_count,
        char_count=record.char_count,
        head=record.head,
        tail=record.tail,
    )
    recall_command = (
        "onecomp-arc-recall --session-id " f"{shlex.quote(session_id)} --obs-id {record.obs_id}"
    )
    citation = re.sub(
        r"\[to recall full output:.*?no output follows\]",
        f"[tool completed; to recall exact output, run `{recall_command}`]",
        citation,
        flags=re.DOTALL,
    )

    if platform == "codex":
        # Codex currently has no updatedToolOutput field.  Its documented
        # PostToolUse block feedback replaces the completed tool result while
        # retaining the tool's side effects.
        return {"decision": "block", "reason": citation}

    replacement = _claude_replacement(response, citation)
    if replacement is None:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "updatedToolOutput": replacement,
        }
    }


def _command_signature(tool_name: str, tool_input: Any) -> str:
    if isinstance(tool_input, dict) and isinstance(tool_input.get("command"), str):
        return tool_input["command"]
    return f"{tool_name}({_lossless_json_text(tool_input)})"


def _lossless_json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _returncode(response: Any) -> int:
    if isinstance(response, dict):
        for key in ("returncode", "return_code", "exit_code", "exitCode"):
            value = response.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return 0


def _claude_replacement(response: Any, citation: str) -> Any:
    if isinstance(response, str):
        return citation
    if isinstance(response, list):
        return [{"type": "text", "text": citation}]
    if isinstance(response, dict) and isinstance(response.get("stdout"), str):
        replacement = dict(response)
        replacement["stdout"] = citation
        if isinstance(replacement.get("stderr"), str):
            replacement["stderr"] = ""
        return replacement
    # Built-in Claude tools require exact output schemas.  Unknown object
    # shapes fail open instead of risking that Claude ignores the replacement.
    return None


def _hook_root(root: Optional[Path | str] = None) -> Path:
    resolved = Path(
        root
        if root is not None
        else os.environ.get(
            "ONECOMP_ARC_HOOK_ROOT",
            str(Path.home() / ".cache" / "onecomp" / "arc-hooks"),
        )
    ).expanduser()
    resolved.mkdir(parents=True, exist_ok=True)
    try:
        resolved.chmod(0o700)
    except OSError:
        pass
    return resolved.resolve()


def _wal_path(session_id: str, root: Optional[Path | str] = None) -> Path:
    digest = hashlib.sha256(session_id.encode("utf-8", errors="replace")).hexdigest()[:32]
    return _hook_root(root) / f"{digest}.wal"


def hook_main() -> None:
    parser = argparse.ArgumentParser(description="ARC PostToolUse hook")
    parser.add_argument("--platform", choices=("claude", "codex"), required=True)
    parser.add_argument("--result-max-chars", type=int, default=500)
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        result = process_post_tool_use(
            payload,
            platform=args.platform,
            result_max_chars=args.result_max_chars,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        # Hook failure is intentionally fail-open: never hide a tool result if
        # persistence or payload parsing failed.
        print(f"onecomp ARC hook skipped: {exc}", file=sys.stderr)
        result = {}
    print(json.dumps(result, ensure_ascii=False))


def recall_main() -> None:
    parser = argparse.ArgumentParser(description="Recall one ARC hook observation")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--obs-id", required=True)
    args = parser.parse_args()
    if not _OBS_ID_RE.fullmatch(args.obs_id):
        parser.error("--obs-id must be 4-40 hexadecimal characters")
    store = ObsStore.from_wal(_wal_path(args.session_id))
    record = store.get(args.obs_id)
    if record is None:
        print(f"Unknown ARC observation id: §{args.obs_id}", file=sys.stderr)
        raise SystemExit(2)
    sys.stdout.write(record.full_content)


__all__ = ["hook_main", "process_post_tool_use", "recall_main"]
