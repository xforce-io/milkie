# s-002 — Inspect a Completed Run (HTML report)

Runnable example for story
[`s-002-inspect-a-completed-run`](../../docs/stories/s-002-inspect-a-completed-run.md).

Demonstrates **`trace report`** — the HTML projection over an event log.
The same data you can read via `milkie trace inspect <runId>` (JSONL) is
rendered into a self-contained HTML file that opens in any browser,
without a server or framework.

This example exists to validate two architectural claims:

1. Visual rendering changes trace **affordance** vs. CLI output alone —
   a folded timeline with click-to-expand entries makes "what did the
   agent do" discoverable in a way `inspect` JSONL is not.
2. The CLI JSON output is rich enough to **fully drive** a UI projection
   (ARCHITECTURE.md `## User-facing surfaces` — UI is a pure projection
   over CLI / SDK output, never a parallel facade).

## Files

```
.milkie/
  agents.json       # manifest
  runs/             # JsonlEventStore base (filled by record.ts)
  last-run.txt      # runId of the most recent recording
agents/
  echo.md
record.ts           # records a sample run with a stub gateway (no API key)
report.sh           # `milkie trace report <runId> > report.html`
report.html         # generated artifact (gitignored)
README.md
```

## Run it

From the repo root, build once so the CLI binary exists:

```bash
$ npm run build
```

Then:

```bash
# 1. Record a sample run (uses a tiny in-process StubGateway, no API key).
$ npx tsx examples/s-002-inspect/record.ts
{
  "runId":  "...",
  "status": "completed",
  "output": "hello, milkie!",
  "eventFile": ".../examples/s-002-inspect/.milkie/runs/....jsonl"
}

# 2. Render the report.
$ ./examples/s-002-inspect/report.sh
wrote report.html for run ...

# 3. Open the report in a browser.
$ open examples/s-002-inspect/report.html
```

## What's in the report

- Run header: agent id, runId, status badge.
- Event timeline: one entry per LLM call / tool call / lifecycle event;
  paired `*.requested` / `*.responded` events collapse to a single entry
  via their `causedBy` chain.
- Click any entry to expand its payload — for LLM/tool entries the raw
  request and response JSON; for lifecycle entries the started/completed
  payload (agent id, goal, parentId, status, last text output).
- Type-filter chips (LLM / tool / lifecycle) at the top.
- Sub-agent runs (when present) nest under their parent as indented
  child timelines — same layout, recursively.

## What this proves

- **CLI ↔ projection parity.** `report.html` consumes nothing beyond
  the data `trace inspect --include-children` already emits; the renderer
  cannot reach into the event store. If the report can show it, the CLI
  can output it.
- **Self-contained artifact.** The HTML file embeds its own raw events
  as `<script type="application/json" id="trace-data">`, making it a
  re-renderable archive: future renderer versions can re-render the
  same file without re-running the agent.

## What this does NOT prove yet

- **Fork / diff / lineage / suite views.** These require Phase 5–6
  capabilities that aren't in code yet; the report scope is intentionally
  limited to the Observable surface.
- **In-flight rendering.** The report is for completed runs. Mid-run
  updates require the Phase 5 in-flight trace query API.

## Related

- Story: [s-002](../../docs/stories/s-002-inspect-a-completed-run.md)
- Spec: [CLI surface](../../docs/superpowers/specs/2026-05-24-cli-surface-design.md)
- Architecture: [`Observable` capability](../../ARCHITECTURE.md#representative-scenarios)
