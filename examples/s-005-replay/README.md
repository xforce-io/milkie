# s-005 — Deterministic Replay

Runnable example for story
[`s-005-deterministic-replay`](../../docs/stories/s-005-deterministic-replay.md).

Demonstrates milkie's **structural replay**: a recorded run can be
re-executed with byte-equivalent state and **zero live LLM calls** —
the response cache (Phase 3) serves every model call from the recorded
event log. The same replay is accessible through two surfaces:

- **SDK**: `Milkie.replay(runId)` in TypeScript
- **CLI**: `milkie trace replay <runId>`

Both produce the same `{ status, output }` because the CLI is a thin
wrapper over the SDK (see CLI surface spec §8).

## Files

```
.milkie/
  agents.json         # manifest declaring the `echo` agent
  runs/               # JsonlEventStore base (runs land here as <runId>.jsonl)
  last-run.txt        # runId of the most recent recording (written by record.ts)
agents/
  echo.md             # the `echo` agent definition (one LLM state, no tools)
record.ts             # records a sample run using an in-process stub gateway
replay-sdk.ts         # replays via SDK
replay-cli.sh         # replays via CLI
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
$ npx tsx examples/s-005-replay/record.ts
{
  "runId":  "8c3a9...",
  "status": "completed",
  "output": "hello, milkie!",
  "eventFile": ".../examples/s-005-replay/.milkie/runs/8c3a9....jsonl"
}

# 2. Replay via SDK.
$ npx tsx examples/s-005-replay/replay-sdk.ts
{
  "via":    "sdk",
  "runId":  "8c3a9...",
  "status": "completed",
  "output": "hello, milkie!"
}

# 3. Replay via CLI — same output, different surface.
$ ./examples/s-005-replay/replay-cli.sh
{"newRunId":"8c3a9...","status":"completed","output":"hello, milkie!"}
```

Both replays produce the same `status` and `output` as the original
record, with **zero live LLM calls** in between — the StubGateway
threw `'exhausted'` if invoked but is never reached on replay.

## What this proves

- **`run` is a first-class artifact.** The event log on disk is the
  authoritative source of truth; you can throw away the runtime state and
  reconstruct it deterministically (ARCHITECTURE.md invariants 1, 8).
- **CLI ↔ SDK parity.** Both surfaces invoke the same underlying
  primitive; same input → same output. The CLI doesn't own its own
  query / replay logic — it routes through the library (ARCHITECTURE.md
  invariant 13).
- **No API key required to run a recorded agent.** Once events are on
  disk, replay is offline and free (CLI surface spec §4.5).

## What this does NOT prove yet

- **Byte-identical replay.** Phase 3 ships structural equivalence
  (status, output, key event sequence). Timestamps and UUIDs are
  re-sampled. The non-determinism log (Phase 4) closes this gap; see
  `ARCHITECTURE.md` Implementation Status.

## Related

- Story: [s-005](../../docs/stories/s-005-deterministic-replay.md)
- Spec: [CLI surface](../../docs/superpowers/specs/2026-05-24-cli-surface-design.md)
- Spec: [Agent registration](../../docs/superpowers/specs/2026-05-24-agent-registration-design.md)
- Architecture: [`Replay` capability](../../ARCHITECTURE.md#representative-scenarios)
