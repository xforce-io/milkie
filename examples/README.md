# Examples

Runnable demos of milkie. Each example is self-contained: it ships an
SDK script (and, where relevant, the equivalent CLI invocation) plus a
README that links back to the story or spec it validates.

| Example | Demonstrates | Surface | API key |
|---|---|---|---|
| [`s-002-inspect`](./s-002-inspect/) | `trace report` — HTML projection over an event log | CLI | not required (stub gateway) |
| [`s-005-replay`](./s-005-replay/) | Deterministic replay — re-run a recorded run with zero live LLM calls, identical output via SDK and CLI | SDK + CLI | not required (stub gateway) |
| [`agent-docs-qa`](./agent-docs-qa/) | A real Q&A agent over a 三国演义 corpus, with live trace observation and skill loading in a local web UI | Web app | **required** (real LLM) |

Start with `s-005-replay` for the core record→replay model, then
`s-002-inspect` to see the same event log rendered as HTML. Reach for
`agent-docs-qa` when you want a full agent + live-trace UI you'd actually
use.

Each example's own README has the exact run commands. All of them assume
you've built the CLI once from the repo root:

```bash
npm run build
```

## Naming

`s-002-` / `s-005-` carry a **story number** that maps one-to-one to a
file under [`docs/stories/`](../docs/stories/) — the prefix is the link
between the example and the story it proves. `agent-docs-qa` has no such
prefix because it validates a design spec rather than a numbered story.
So the naming difference is intentional, not drift.

## Fixtures vs. runtime state

`s-002-inspect` and `s-005-replay` **commit** their `.milkie/runs/*.jsonl`
and `.milkie/last-run.txt`. That recorded run is a fixture: replay and the
HTML report are projections over it, so it has to be frozen and versioned,
not regenerated on demand. Re-running `record.ts` simply produces a fresh
run alongside it.

Everything that is genuine runtime state is gitignored: the SQLite working
state (`state.sqlite*`, ignored for all examples via
[`examples/.gitignore`](./.gitignore)), and — for `agent-docs-qa`, whose
runs are live and not fixtures — its `.milkie/runs/` and `.milkie/objects/`.
