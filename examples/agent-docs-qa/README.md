# agent-docs-qa — 三国 Q&A with live trace observation

Runnable example: a Q&A agent over a vendored 三国演义 corpus, with a small
local web server that exposes **live trace observation** and **skill-loading
demonstration**.

Design spec: [docs/superpowers/specs/2026-05-24-agent-docs-qa-example-design.md](../../docs/superpowers/specs/2026-05-24-agent-docs-qa-example-design.md)

## What this example demonstrates

1. **A real agent you'd actually use** — ask questions about 三国演义,
   get cited answers. Not a toy.
2. **Live trace observation** — UI shows the agent's full reasoning
   trajectory (grep, read, LLM call/response, lifecycle) as it happens.
3. **Skill loading (progressive disclosure)** — when you ask "你确定吗?",
   the agent calls `skill_request("verifier")` mid-conversation; the next
   turn loads a stricter verification skill into the context. The UI
   highlights the moment of skill load.
4. **Conversation browsing** — past conversations persist as JSONL on
   disk; the UI picker lets you revisit them (read-only).

## Setup

You need an OpenAI-compatible LLM endpoint. Set the API key in your env:

```bash
export OPENAI_API_KEY=sk-...
# or your provider's equivalent
```

The default agent uses `gpt-4o-mini` via the `openai-compatible` adapter.
To swap providers, edit `agents/sanguo-researcher.md` `model:` section.

## Run

```bash
# From repo root:
npm run build

cd examples/agent-docs-qa
npx tsx server.ts
# → agent-docs-qa playground at http://localhost:7878
```

Open `http://localhost:7878` in a browser.

## Try these questions

- 赤壁之战双方主帅是谁？
- 诸葛亮第一次出场是在哪一回？
- 关羽华容道为什么放走曹操？
- 桃园结义里三兄弟是谁？

After any answer, type **"你确定吗？"** (or "再确认下" / "真的吗")
to trigger the verifier skill load. Watch the trace timeline: a yellow-
highlighted `tool.requested · skill_request` entry appears with the tooltip
`Skill load requested: verifier`. The next LLM call's system prompt will
include the verifier instructions — click the `llm.requested` entry to
inspect the raw payload and see for yourself.

## What's in the corpus

Five chapters of 三国演义 from Wikisource (public domain — 罗贯中, c.1330-1400):

- 第一回 桃园三结义
- 第三十七回 三顾茅庐
- 第四十九回 赤壁借东风
- 第五十回 华容道
- 第六十六回 单刀赴会

Files vendored to `corpus/` directory, **frozen at example creation time** —
they do NOT track main repo docs changes. To use your own corpus, replace
files in `corpus/` (filenames don't matter; agent uses `list_dir` to
discover them).

## Architecture (one diagram)

```
browser (vanilla JS)         Node http server           Milkie SDK
─────────────────────       ───────────────────         ──────────
chat input ─POST /chat───►  milkie.invoke({          RecordingIOPort
                              contextId, input })    ↓
trace area ◄─SSE stream───  BroadcastingEvent       events flow into
   (auto-renders             Store: write to        JsonlEventStore
    incoming events)         JSONL + broadcast      (.milkie/runs/)
                             to subscribers
conv picker ◄─/convs ────   scan .milkie/runs/
                             group by contextId
```

## Endpoints

- `POST /chat` — body `{ input, contextId? }` → invoke agent; returns `{ runId, contextId, status, output }`
- `GET /conversations` — list of `{ contextId, agentId, startedAt, status, runIds, eventCount }`
- `GET /conversation/:id/events` — full events array, time-ordered (404 if unknown)
- `GET /conversation/:id/stream` — SSE: past events + live append subscription; empty close if unknown contextId
- `GET /` — single-page UI

## File layout

```
.
├── agents/sanguo-researcher.md    # AgentConfig: base + verifier skill
├── corpus/                        # vendored chapters (5 .txt files)
├── tools/corpus-tools.ts          # sandboxed list_dir / read_file / grep
├── trace/
│   ├── broadcast-event-store.ts   # IEventStore + per-context pub/sub
│   └── conversation-scanner.ts    # group runIds by contextId from disk
├── server.ts                      # http + SSE
├── public/index.html              # vanilla UI (chat + trace + payload)
└── __tests__/                     # 36 tests: unit + integration + automated e2e
```

## Tests

```bash
# From repo root
npx jest examples/agent-docs-qa/__tests__/
# → 36 tests pass
```

The automated e2e in `__tests__/server.test.ts`
(`describe('e2e: skill loading through full chat flow')`) is the critical
assertion: it uses a RecordingStubGateway to fire a real
`skill_request('verifier')` tool call mid-invoke and verifies that the
**next LLM request's system prompt** contains the verifier instructions
(literal phrase `'你已进入 verifier 模式'`). This proves the skill-loading
mechanism is wired end-to-end through `Milkie.invoke` →
`applyPendingSkills` → context epoch rebuild → next LLM call. Manual UI
testing can confirm visual correctness but this test is what guards
correctness in CI.

## Manual smoke checklist

After `npx tsx server.ts`, open `http://localhost:7878` and verify:

1. **Cold start**: page loads, dropdown shows "(new conversation)", input enabled.
2. **First question**: type "赤壁之战双方主帅是谁？" → submit. Trace timeline
   fills as grep/read/LLM events stream in. Final answer mentions 周瑜/曹操.
3. **Conversation continuity**: type "他们最后谁赢了？" → agent answers using
   prior context.
4. **Skill loading**: type "你确定吗？" → trace shows a yellow-highlighted
   `tool.requested · skill_request` entry (tooltip: `Skill load requested: verifier`);
   the following `llm.requested` payload's `system` field includes verifier instructions.
5. **Conversation switching**: click "+ new chat" → URL `?context=` drops →
   ask different question. Then open the picker — both conversations listed;
   click the first one — chat + trace restore from past events; input is
   disabled with "此对话已结束".
6. **Payload click**: click any trace entry → JSON payload renders in dark
   panel at bottom. For an `llm.requested` after skill load, the `system`
   field includes verifier text.

## What this does NOT do (intentional scope)

- **Resume past conversations** — past conversations are read-only; clicking
  one lets you browse but not continue. Use "+ new chat" to start fresh.
- **Multi-trajectory comparison view** — the data model + UI are ready for
  `?compare=A,B` (Phase 5 fork/diff substrate), but the comparison view
  itself isn't built.
- **Multi-user / auth** — single-process localhost only.
- **Persistent active state across server restart** — server uses
  `MemoryStore`; in-flight conversations are lost on restart. Trace JSONLs
  remain on disk and stay browseable.
- **Custom corpus loading at runtime** — corpus path is hardcoded to
  `./corpus/`; replace files in that dir to change content.

## Known substrate limitations (addressed in pending design)

This example is built on milkie's current `ContextLayer`, which has several
architecturally-incomplete behaviors. The example honestly exposes them
rather than papering over them:

- **No skill release path** — the agent can `skill_request` to load `verifier`,
  but has no way to unload it. Once loaded, verifier stays in the system
  prompt for the remainder of the conversation. Progressive disclosure is
  one-way today.
- **scratchpad and history are conflated** — the ReAct loop's
  `assistant tool_use` / `tool` messages accumulate in the same `history`
  array as cross-turn `(user, finalAssistant)` pairs. Across many turns the
  intermediate tool-call noise crowds the LLM's context.
- **Tool results have no upper bound** — `read_file` returns full chapter
  bodies (10–20KB each) verbatim into history; nothing in the substrate
  enforces a per-tool size policy.

All three are addressed in:
**`docs/superpowers/specs/2026-05-25-context-region-substrate-design.md`**

After that substrate work lands (region abstraction + lifetime declaration +
turn-end crystallization + `ToolResultStrategy`), this example's
frontend + `agents/sanguo-researcher.agent.md` will be updated to demonstrate
the full *load → use → auto-release at turn end* cycle and per-tool result
shaping.

## Related

- Story: [s-002](../../docs/stories/s-002-inspect-a-completed-run.md)
  (the static report this builds on)
- Story: [s-010](../../docs/stories/s-010-skill-versioned-load-and-ab-experiment.md)
  (the skill-loading capability this exercises end-to-end)
- Spec: [Context Region Substrate](../../docs/superpowers/specs/2026-05-25-context-region-substrate-design.md)
  (the design that addresses the limitations listed above)
- Architecture: [ARCHITECTURE.md](../../ARCHITECTURE.md) — "Reference UI projection"
