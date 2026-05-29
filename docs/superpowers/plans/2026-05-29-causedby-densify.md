# causedBy 加密（4 条边）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 trace 事件织入 4 条 causedBy 边，把今天的「配对边集合」补成可顺向回溯的因果图：
`tool.requested ← llm.responded`、`llm.requested ← 上个终结者`、`fsm.transition ← 触发事件`、
`region.added ← boundary.applied`。replay byte-identical 不变。

**Architecture:** 引入 per-run 的纯状态对象 `CausalCursor`（3 字段），注入同一 run 的
`RecordingIOPort` 与 `AgentRuntime`。各 emit 点在**同步时刻**读 cursor 盖 causedBy、emit 后写 cursor
（铁律：绝不在延迟的 `enqueueTraceWrite` 闭包内读写）。3 条边本地可解，唯 `fsm.transition` 经 cursor
跨文件读 `lastIoEventId`。trace 事件 id 是裸 uuid、不进 nondet 缓存 → causedBy 对 replay 零影响。

**Tech Stack:** TypeScript (ESM, `.js` import 后缀)、jest（`npx jest <file> -t "<name>"`）。

**设计依据：** `docs/superpowers/specs/2026-05-29-causedby-densify-design.md`

---

## File Structure

- `src/trace/CausalCursor.ts` — 新建，纯状态对象。
- `src/trace/RecordingIOPort.ts` — 构造函数加可选 `cursor?`；`attach` / `invokeLLM` / `invokeTool`
  读写 cursor、盖边 1（tool.requested）、边 2（llm.requested）。
- `src/runtime/AgentRuntime.ts` — opts 加可选 `causalCursor?`；`onTransitionCallback` 盖边 3；
  `emitBoundaryApplied` 记 `lastBoundaryId`、`emitRegionDelta` 盖边 4。
- `src/runtime/Milkie.ts` — `wrapIOPort` 加 cursor 参数；`run`/`resume` 建 cursor 喂双方；
  `buildMakeChildPort` 建子 cursor、返回值带 `cursor`。
- `src/__tests__/CausedByGraph.test.ts` — 新建，4 条边 + 图连通 + 无孤儿 + cursor 缺省。
- `src/__tests__/Replay.test.ts` / `s-005` — 回归（不改，跑通即可）。

---

## Task 1: CausalCursor + 边 1/边 2（RecordingIOPort，纯单元）

**Files:**
- New: `src/trace/CausalCursor.ts`
- Modify: `src/trace/RecordingIOPort.ts`（constructor `:58-64`、`attach :117`、`invokeLLM :141`、`invokeTool :173`）
- Test: `src/__tests__/CausedByGraph.test.ts`（新建，先覆盖边 1/边 2）

- [ ] **Step 1 写失败测试**：用 `ExecInnerPort`（仿 `RecordingIOPort.metadata.test.ts` 的 harness，
  `invokeLLM` 可配置返回带 toolCalls 的 response）驱动一次「llm.responded(toolCalls) → tool.requested
  → tool.responded → 第二次 llm.requested」序列，注入 `new CausalCursor()`，断言：
  - `tool.requested.causedBy === <llm.responded.id>`（边 1）
  - 首个 `llm.requested.causedBy === <agent.run.started.id>`（边 2 seed）
  - 第二个 `llm.requested.causedBy === <上一条 tool.responded.id>`（边 2）
- [ ] **Step 2 实现 CausalCursor**：`src/trace/CausalCursor.ts`，3 个可选字段
  `lastIoEventId` / `lastLlmRespondedId` / `lastTerminatorId`（见 spec §CausalCursor）。
- [ ] **Step 3 接 RecordingIOPort**：
  - constructor 末位加可选 `cursor?: CausalCursor`。
  - `attach`：emit 后 `cursor?.lastTerminatorId = <id>`。
  - `invokeLLM`：`llm.requested` 盖 `causedBy = cursor?.lastTerminatorId`；emit 后
    `cursor.lastIoEventId = reqId`。`llm.responded` emit 后 `cursor.lastLlmRespondedId = id`、
    `cursor.lastIoEventId = id`。
  - `invokeTool`：`tool.requested` 盖 `causedBy = cursor?.lastLlmRespondedId`；emit 后
    `cursor.lastIoEventId = reqId`。成功 / 错误 `tool.responded` emit 后
    `cursor.lastTerminatorId = id`、`cursor.lastIoEventId = id`。
    **此处加强制注释**（spec §F2 实现注释要求）：并行批下取最后完成的 tool.responded 是有意、对
    正确性/replay 无害。
  - cursor 缺省 → 全部 `causedBy` 保持 `undefined`，行为不变。
- [ ] **Step 4 cursor 缺省回归测试**：不注入 cursor 跑既有序列，断言无新 causedBy、不抛错。
- [ ] **Step 5** 全测试过；`npx jest CausedByGraph`。

## Task 2: 边 3（fsm.transition）+ 边 4（region.added）（AgentRuntime）

**Files:**
- Modify: `src/runtime/AgentRuntime.ts`（opts/字段、`onTransitionCallback :160`、
  `emitRegionDelta :436-457`、`emitBoundaryApplied :590-612`、私有字段 `lastBoundaryId`）
- Test: `src/__tests__/CausedByGraph.test.ts`（扩展，用真实 `Milkie` 端到端跑一次带工具 + turn-end 的 run）

- [ ] **Step 1 写失败测试**：经 `Milkie`（StubGateway 喂固定 response + 一个会 `ctx.emit` 的工具）跑一
  次多轮 run 到 turn-end，断言：
  - 工具触发的 `fsm.transition.causedBy === <触发它的 tool.responded.id>`（边 3）
  - turn-end 的 `region.added.causedBy === <对应 context.boundary.applied.id>`（边 4）
  - `agent-set` 初始 region（header）`causedBy` 缺省（允许的例外）
- [ ] **Step 2 接 cursor 到 AgentRuntime**：opts 加可选 `causalCursor?: CausalCursor`，存私有字段；
  `factory`（`:144`）建子 runtime 时不直接传（子 cursor 走 makeChildPort，见 Task 3）。
- [ ] **Step 3 边 3**：`onTransitionCallback` 内**同步**读 `this.causalCursor?.lastIoEventId`，作为
  `fsm.transition` 事件的 `causedBy`，再 enqueue（值已捕获，落盘晚无妨）。
- [ ] **Step 4 边 4**：`emitBoundaryApplied` 里把 `context.boundary.applied` 事件 id 存
  `this.lastBoundaryId`（生成 id 时同步赋值）；`emitRegionDelta` 构造 `region.added`（仅 `added`）
  时盖 `causedBy = this.lastBoundaryId`。`removed` 不盖；`lastBoundaryId` 为空（agent-set）则缺省。
- [ ] **Step 5** `npx jest CausedByGraph` 全过。

## Task 3: Wiring（root + child cursor 注入）

**Files:**
- Modify: `src/runtime/Milkie.ts`（`wrapIOPort :63`、`run :210`、`resume :291`、`buildMakeChildPort :70`）
- Modify: `src/runtime/AgentRuntime.ts`（`makeSubAgentTool :240-262` 接 `built.cursor`；`MakeChildPort` 返回类型加 `cursor`）

- [ ] **Step 1 root**：`run`/`resume` 内 `const cursor = new CausalCursor()`；`wrapIOPort` 加第三参
  `cursor` 透传给 `RecordingIOPort`；同一 cursor 进 AgentRuntime opts `causalCursor`。
- [ ] **Step 2 child**：`buildMakeChildPort` 内 `const cursor = new CausalCursor()` 注入子 port，
  返回 `{ port, finish, cursor }`；`MakeChildPort` 类型同步加 `cursor`。
- [ ] **Step 3 child runtime**：`makeSubAgentTool` 拿 `built.cursor` 传子 AgentRuntime `causalCursor`。
- [ ] **Step 4 子 agent 隔离测试**：基于 s-007 harness（父含 sub-agent 工具），断言父子各自 4 条边
  自洽、cursor 不串。

## Task 4: 验收 + replay 回归

- [ ] **图连通**：任取 `tool.requested` 顺 causedBy 走到 `agent.run.started`（验收 1）。
- [ ] **boundary 定位**：boundary 触发的 `region.added` 能定位到 `context.boundary.applied`（验收 2）。
- [ ] **无孤儿**：除 `agent.run.started` 与 agent-set `region.added` 外均有 causedBy（验收 3，两类例外见 spec）。
- [ ] **replay**：`npx jest Replay` + `examples/s-005-replay` 全过，byte-identical 不变（验收 4）。
- [ ] **全量**：`npm test` 绿。

---

## 提交节奏（按仓库惯例，等用户拍板再 commit）

1. `docs(trace): design spec + plan for causedBy densify (#30)`
2. `feat(trace): CausalCursor + tool.requested/llm.requested causedBy (#30)`（Task 1）
3. `feat(trace): fsm.transition + region.added causedBy (#30)`（Task 2）
4. `feat(trace): wire CausalCursor through root + child ports (#30)`（Task 3）
5. `test(trace): causedBy graph connectivity + replay regression (#30)`（Task 4）

## 不在本 plan（spec deferral）

- 统一 `emitTraceEvent` helper 收口；causedBy 图可视化 / explainer（#32-36）；并行批多终结者精确归因。
