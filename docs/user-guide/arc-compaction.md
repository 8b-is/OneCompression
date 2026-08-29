# Addressable context compaction

`onecomp.compaction` reduces an agent's model-visible history while preserving
large tool results in an exact, addressable archive. The production
`formal-v1` profile is dependency-free and can run independently of model
quantization, distillation, or a GPU.

## Quick start

```python
from onecomp import CompactionConfig, CompactionRunner

config = CompactionConfig.persistent(
    ".onecomp/compaction",
    keep_recent=2,
    summary_max_chars=12_000,
    max_recall_units=2_048,
)

with CompactionRunner(config) as compactor:
    record = compactor.record_observation(
        command="pytest -q",
        content=large_test_output,
        source_key="tool-call-0182",
        returncode=1,
        tags=["test"],
    )
    messages.append(compactor.make_observation_message(record))
    messages[:] = compactor.enforce_char_budget(messages, 24_000)

    # Use the host tokenizer when a character envelope is not authoritative.
    messages[:] = compactor.enforce_token_budget(messages, 6_000, count_message_tokens)
```

`source_key` is the host's tool-call or event identifier. A retry with the same
key and identical data returns the same observation ID. Reusing the key with
different data fails. Two different calls receive different IDs even when
their command and output are byte-for-byte identical. If the host does not
provide a stable occurrence identifier, omit `source_key`: OneComp preserves
each call as a distinct occurrence instead of guessing that equal content is a
retry.

## Exact recovery

The model-facing grammar is deliberately strict:

```text
_catalog [start|cursor] [limit]
_catalog find <command-or-tag>
_recall_meta §<id>
_recall §<id>
_recall_chunk §<id> <index>
_recall_range §<id> <start>:<end>
```

A full recall that cannot fit `max_recall_units` is rejected. The caller must
use deterministic chunks or a half-open character range. Adjacent chunks
reconstruct the original Unicode string exactly; each catalog row carries the
body SHA-256 without exposing the body.

```python
response = compactor.handle_action(f"_recall_chunk §{record.obs_id} 0")
assert response is not None and response.exact
print(response.body)
```

Previews inside citations are explicitly non-authoritative. Copy exact values
only from a successful recall. `citation_head_chars` and
`citation_tail_chars` independently bound the two previews; short bodies are
shown once rather than duplicated where those windows overlap.

Formal archives maintain an exact-filtered SQLite trigram index when the local
SQLite build provides FTS5. Progress checks and body-substring lookups use the
index for literals of at least three characters and retain a literal scan
fallback, so acceleration does not change exact match semantics.

Exact range and default character-chunk recalls use a sparse block index of
UTF-8 byte offsets and per-block hashes. The canonical body remains stored once;
only overlapping blocks are read and verified. Archives created before this
index are verified and backfilled transparently on first partial recall.

## Guard the final provider request

Compacting an intermediate list is not enough to prove a context bound. Tool
schemas, role wrappers, images, and provider serialization can add tokens. Put
the authoritative counter at the final dispatch boundary:

```python
from onecomp.compaction.adapters import GuardedProviderAdapter

adapter = GuardedProviderAdapter(
    compactor,
    count_request=provider_token_counter,
)
result = adapter.dispatch(
    {"model": model_name, "messages": messages, "tools": tools},
    provider_client,
    compact_to_chars=24_000,
)

audit = adapter.audit()
assert audit.checked_requests == audit.dispatched_requests
```

Configure `context_window`, `completion_reserve`, and a conservative
`serialization_margin` in `CompactionConfig`. An oversized final request is
rejected before the provider function is called. The adapter also replaces the
host-owned transcript after successful compaction; retaining only a compacted
copy while leaving the private history untouched is not supported.

## Standalone CLI

```bash
onecomp-arc --config .onecomp/compaction/config.json init

# Agent-safe command boundary: short output passes through, verbose output is archived.
onecomp-arc --config .onecomp/compaction/config.json run -- pytest -q

onecomp-arc --config .onecomp/compaction/config.json record \
  --command "pytest -q" --content-file test-output.txt \
  --source-key tool-call-0182 --tag test

onecomp-arc --config .onecomp/compaction/config.json catalog --query pytest
onecomp-arc --config .onecomp/compaction/config.json recall OBS_ID
onecomp-arc --config .onecomp/compaction/config.json recall OBS_ID --kind chunk --chunk-index 0
onecomp-arc --config .onecomp/compaction/config.json doctor
```

Exact full/chunk/range bodies are written directly to stdout for lossless shell
redirection. Metadata is the safe default. `export` redacts bodies and
free-form host extras unless
`--include-bodies` is explicitly supplied. File exports and legacy migrations
stream bounded batches instead of materializing every archived body at once.

`run` executes an argv list without a shell. It preserves the child exit status
and combines stdout/stderr in execution order. Long UTF-8 output becomes a
bounded citation only after its archive transaction commits. Archive failure,
non-UTF-8 output, or output above `--max-output-bytes` fails open and returns the
original bytes. Interactive terminals stream command progress to stderr while
keeping stdout reserved for the final citation. Use `--tee` to enable this in
CI, or `--no-tee` to keep stderr quiet. After `--max-output-bytes` is crossed,
the original stream is forwarded immediately without growing the temporary
capture. Use `onecomp-arc guide --host codex` to print concise project
instructions for a coding agent. `--config` is accepted before or after the
subcommand. Configuration and redacted export files use private, atomic writes;
an existing destination is not overwritten unless the corresponding force
option is explicit.

`init --durability full` is the default and fsyncs body transactions. Select
`--durability normal` only when higher ingestion throughput is worth relaxing
power-loss durability; SQLite transaction and process-crash recovery remain.
Recall counters are telemetry rather than recovery data and are flushed in a
batch at status, report, export, and clean close boundaries.

To migrate the earlier JSONL format without changing old citations:

```bash
onecomp-arc --config .onecomp/compaction/config.json migrate old.wal --dry-run
onecomp-arc --config .onecomp/compaction/config.json migrate old.wal
```

## Agent-host integration

### Claude Code hook

Install the `PostToolUse` hook using the environment's entry point:

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Bash|Read|Grep|Glob",
      "hooks": [{
        "type": "command",
        "command": "/path/to/venv/bin/onecomp-arc-hook --platform claude"
      }]
    }]
  }
}
```

Claude output replacement preserves known built-in output shapes and fails open
for unknown structured or multimodal shapes. This avoids converting a valid
host payload into a text-only shape that the host cannot apply.

Set `ONECOMP_ARC_HOOK_ROOT` to a dedicated private directory. A tool result is
replaced only after its SQLite transaction commits. The generated citation
contains a bounded `onecomp-arc-recall` command addressed through the same
Python environment as the hook, so it does not depend on the agent shell's
`PATH`. Recall opens the committed archive read-only and therefore also works
when the agent sandbox can read, but cannot mutate, the hook cache. Recall
telemetry is intentionally not persisted at this read-only boundary. Use
`--max-archive-bytes BYTES` when the hook process itself must enforce a per-session
UTF-8 body quota.

### Codex hook and command boundary

The current Codex `PostToolUse` contract replaces a completed tool result with
the hook `reason` when `decision` is `block`. OneComp exposes this behavior as
`--codex-result-mode replace`, the default. The original result commits before
the bounded citation is returned. The standalone boundary remains useful for
older host versions and commands outside the hook contract:

```bash
onecomp-arc --config .onecomp/compaction/config.json run -- pytest -q
onecomp-arc --config .onecomp/compaction/config.json run -- rg pattern src tests
```

`context` is an explicit archival-only compatibility mode that retains native
output. `block` aliases `replace` at the current wire layer, while `stop` uses
the universal continuation contract. Validate host upgrades against the
checked-in contract test before relying on replacement.

### OpenCode

Generate a typed plugin and place it in the host's plugin directory:

```bash
onecomp-arc --config .onecomp/compaction/config.json plugin \
  --host opencode --output .opencode/plugins/onecomp-compaction.ts
```

The plugin keys persistence by `sessionID` and `callID`, replaces
`output.output` in `tool.execute.after`, and adds recall guidance during
`experimental.session.compacting`.

### OpenHands SDK

The optional adapter emits a native `Condensation`, so forgotten events are
actually removed from the model view. It registers paired recall and catalog
tools backed by the same formal archive. Install it with
`pip install onecomp[openhands]`; the SDK is intentionally not a OneComp core
dependency. The adapter is exercised against the SDK's immutable `View`,
manipulation boundaries, rolling-condenser lifecycle, and native tool schema.
A recalled observation retains its original occurrence identity when condensed
again, so the recall tool does not recursively create duplicate archive rows.

### MCP

`onecomp-mcp` starts in automatic protocol mode and exposes authenticated
formal sessions by default. MCP provides storage and recall, but a stock MCP
tool cannot mutate its host's private transcript. Automatic reduction still
requires a condenser, hook, or custom host that applies the returned message
list. See the [MCP guide](mcp-server.md).

## Single-run evidence

```python
from onecomp.analyzer.compaction import (
    build_run_report,
    save_run_report,
    visualize_run,
)

report = build_run_report(compactor, run_name="repository agent")
save_run_report(report, "results/run-report.json")
visualize_run(report, "results/run-dashboard")
```

Reports and plots include operation timing, compaction trajectory, archive
sizes, recall mix, and checksum coverage. Observation bodies and free-form
extras are excluded. The dashboard uses a white, print-friendly run view with
bounded labels and high-cardinality aggregation. Use the public deterministic
benchmark for comparable measurements:

```bash
python benchmark/compaction/run_benchmark.py \
  --output-dir results/compaction/formal-v1
```

## Guarantees and qualification boundary

`formal-v1` provides the following when its integration contract is followed:

- append-only occurrence identity with fixed-width IDs,
- idempotent host-event retries and conflict detection,
- transactional body/ID commit and crash-safe SQLite WAL persistence,
- indexed literal progress checks with exact post-filtering and a portable fallback,
- exact full, range, text-chunk, and token-ID reconstruction,
- complete body-free catalog pagination,
- bounded model-facing recall responses,
- an auditable final provider-request guard,
- body-free status, report, and visualization telemetry.

The following remain deployment responsibilities:

- choose an authoritative tokenizer and accurate serialization margin,
- secure the archive directory, backups, and process account,
- encrypt storage at the filesystem or volume layer when required,
- define retention/deletion policy outside the append-only archive,
- validate each host version's hook/condenser contract,
- account for multimodal payloads in the final provider counter.

The compatibility-only `experimental-v1` profile preserves earlier behavior
but does not carry the formal guarantees above.
