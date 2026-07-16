# ARC Context Compaction

OneComp ARC (Addressable Recall for Compaction) stores tool output losslessly,
replaces old output with short citations, and restores exact output by
`obs_id`. It is framework-neutral: the host that owns the conversation must
apply the compacted history before sending the next model request.

## Common integration

For each agent step:

1. Store every tool result with `ArcEngine.record_observation`.
2. Add the returned `_obs_id` to the host's tool-result message.
3. Call `ArcEngine.compact` or `enforce_token_budget` before the provider request.
4. Expose recall and, when strict caps can hide command names, the bounded
   observation index to the model.

MCP hosts can use the equivalent `compaction_*` tools from
`onecomp-mcp`. MCP alone cannot rewrite a host's private transcript; use a
host condenser or output hook for automatic reduction.

## OpenHands SDK

The optional adapter emits OpenHands' native `Condensation`, so forgotten
events are actually removed from the model view:

```python
from openhands.sdk import Agent, Tool
from onecomp.compaction.adapters.openhands import OpenHandsArcCondenser

condenser = OpenHandsArcCondenser(max_size=120, max_tokens=100_000)
agent = Agent(
    llm=llm,
    condenser=condenser,
    tools=[
        Tool(name="TerminalTool"),
        Tool(name="OpenHandsArcRecallTool", params={"runtime_id": condenser.runtime_id}),
        Tool(name="OpenHandsArcListTool", params={"runtime_id": condenser.runtime_id}),
    ],
)
```

Use a private `wal_path` when the agent must resume in another process.

## Claude Code

Configure a `PostToolUse` hook. Long supported tool outputs are replaced by a
bounded citation; `onecomp-arc-recall` prints the exact stored output.

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Bash|Read|Grep|Glob",
      "hooks": [{
        "type": "command",
        "command": "/path/to/.venv/bin/onecomp-arc-hook --platform claude"
      }]
    }]
  }
}
```

Set `ONECOMP_ARC_HOOK_ROOT` to a private directory. Structured tool results
with unknown schemas pass through unchanged.

## Codex

Configure the same hook with the Codex platform:

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "^Bash$",
      "hooks": [{
        "type": "command",
        "command": "/path/to/.venv/bin/onecomp-arc-hook --platform codex"
      }]
    }]
  }
}
```

Codex uses PostToolUse feedback replacement rather than Claude's
`updatedToolOutput`. Coverage depends on the tool type; this hook does not
rewrite every private historical item. MCP tools can still provide explicit
record, index, and recall operations.

## Measured result

Measured on 2026-07-16 using 18 real text artifacts from the non-git
`ichikawa_works` validation repository. The API comparison used OpenAI
Responses with `gpt-4.1-mini`; the key was supplied only through the temporary
process environment and was not stored.

| Measurement | Without ARC | With ARC | Reduction / result |
|---|---:|---:|---:|
| Context characters | 267,307 | 26,632 | 90.037% |
| Local `o200k_base` tokens | 69,159 | 8,945 | 87.066% |
| End-to-end API input tokens, 3 queries | 208,355 | 95,036 | 54.4% |
| API input tokens excluding cached input | 208,355 | 39,996 | 80.8% |
| Exact recall checks | — | 18/18 | 100% |
| API answer accuracy | 3/3 | 3/3 | preserved |

The API ARC runs used an index lookup followed by exact recall when a strict
summary cap left an old handle without its command/path mapping.

See the [detailed MCP integration guide](mcp-server.md) for WAL, telemetry,
provider-specific message shapes, and custom host integration.
