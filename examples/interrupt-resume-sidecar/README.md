# interrupt / resume sidecar (#85)

A minimal, deterministic milkie sidecar that demonstrates **cross-process
interrupt/resume over HTTP** — the pattern alfred uses to drive a milkie run
from an external (Python) process.

## Why a sidecar

alfred is a Python process and cannot embed the Node milkie SDK, so it talks to
a milkie sidecar over HTTP. The cross-process boundary is the HTTP layer; inside
the sidecar, `milkie.interrupt(contextId)` / `milkie.resume(...)` are ordinary
in-process SDK calls. Because the sidecar is a single Node process, an interrupt
request that arrives while a run is in flight is serviced on the same event loop,
and the running `AgentRuntime` observes the signal at its next yield point
(turn / tool boundary).

## Endpoints

| Method & path | Body | Response |
|---|---|---|
| `GET /health` | — | `{ ok: true }` |
| `POST /chat` | `{ contextId, goal?, input? }` | `202 { contextId, accepted: true }` — starts a run, returns immediately |
| `GET /status?contextId=…` | — | `{ state: 'running'\|'interrupted'\|'completed'\|'error'\|'unknown', steps }` |
| `POST /interrupt` | `{ contextId }` | `{ signaled: true }` |
| `POST /resume` | `{ contextId, input? }` | `{ status, output }` |

## Run it

```bash
PORT=8090 npx tsx examples/interrupt-resume-sidecar/main.ts
# prints: SIDECAR_READY 8090   (PORT unset/0 → OS picks a free port)
```

Drive it (the "alfred" side):

```bash
curl -XPOST localhost:8090/chat      -d '{"contextId":"c1","input":"start"}'
curl     "localhost:8090/status?contextId=c1"          # watch steps climb
curl -XPOST localhost:8090/interrupt -d '{"contextId":"c1"}'   # run stops → interrupted
curl -XPOST localhost:8090/resume    -d '{"contextId":"c1","input":"continue"}'  # → completed
```

The baked-in agent loops a `work_step` tool `STEPS` times (default 8, `STEP_MS`
ms each) using a local deterministic gateway — no API keys required. The
gateway's step counter and the executed-step set live in the sidecar process,
so a run that is interrupted and later resumed continues from where it stopped
rather than restarting.

## Deployment note (cross-process state)

This demo keeps the run, the interrupt, and the resume **in one sidecar process**
(alfred only triggers them over HTTP), so an in-memory store suffices. If you
split the run and the interrupt across *different* processes (e.g. multiple
sidecar workers), the interrupt flag must live in a shared store — use
`SQLiteStore` (same host) or `RedisStore` (cross-host); `MemoryStore` is
single-process only.

## Tests

- `tests/e2e/interrupt-resume-sidecar.e2e.test.ts` — in-process: real
  `http.Server` + `fetch` over TCP.
- `tests/e2e/interrupt-resume-crossproc.e2e.test.ts` — true cross-process: spawns
  this sidecar via `npx tsx main.ts` and drives it over HTTP.
