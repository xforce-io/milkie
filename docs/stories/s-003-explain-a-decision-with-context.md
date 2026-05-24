---
id: s-003
title: Explain an agent decision with its full context
status: active
kind: scenario
subsystems:
  - agent-trace
capability: explainability
requires:
  - Trajectory observability
  - Agent Trace event log (basic)
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-003-explain-a-decision-with-context.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-trace
---

## 场景叙事

开发者在浏览一个 run 时盯住了某一步决策（"为什么这一轮调了 tool X / 输出了这句话"），想**展开那一瞬间的完整推理材料**：发给 LLM 的 prompt（包含 system、history、可调用工具）、LLM 返回的 response（text + tool_use blocks）、这次决策**导致的**后续动作（tool calls）。

回答的是 "why did the agent decide this"——单点深挖、材料化推理上下文。和 [s-004](./s-004-lineage-from-artifact-to-source.md) 的反向 lineage 不同（后者从 artifact 反推来源）。

这个 story 不依赖 LLM；用 stub gateway 喂固定响应，目的是验证"按决策点拼回上下文"的查询路径，不是 LLM 行为。

## 关键交互流

```
test
  ├─ 构造 milkie + TrajectoryStore + MemoryEventStore
  ├─ 注册 2-state FSM（plan → close）：plan 状态会触发一次 web_search tool
  └─ milkie.invoke(...) → result.agentRunId

explain a decision (无 LLM 调用)
  ├─ 取 eventStore.readByRunId(runId) 全部事件
  ├─ 选定 "第一次 LLM 决策"：events.find(e => e.type === 'llm.requested')
  ├─ 展开它的 payload.request
  │    · request.messages：当时的 system + history（== working context 投影）
  │    · request.tools：那一刻可调用的能力列表
  ├─ 取配对的 llm.responded（causedBy === 该 requested.id）
  │    · response.content：LLM 输出（含 tool_use blocks）
  │    · response.toolCalls：结构化工具调用清单
  ├─ 取这次决策"引发的" tool 事件：events.filter 该 LLM 后紧邻的 tool.requested
  │    · 验证 toolName 与 LLM response.toolCalls 一致
  │    · 取配对的 tool.responded（causedBy === tool.requested.id）→ 看 output
  └─ 拼出"那一瞬间"的完整材料：prompt + response + 触发的 tool I/O
```

## 验收准则

- [ ] `milkie.invoke(...)` 返回 `status: 'completed'`
- [ ] `eventStore.readByRunId(runId)` 至少包含 1 个 `llm.requested` 和 1 个配对的 `llm.responded`
- [ ] 选定的 `llm.requested` 事件的 `payload.request.messages` 数组非空，且包含 `role: 'user'` 消息
- [ ] `payload.request.tools` 非空数组，且包含被注册的 `web_search` 工具名
- [ ] 存在恰好一个 `llm.responded` 事件满足 `responded.causedBy === requested.id`
- [ ] 该 `llm.responded.payload.response.toolCalls` 数组中至少有一个 `name === 'web_search'`
- [ ] 在该 `llm.responded` 之后存在 `tool.requested`，其 `payload.toolName === 'web_search'`，且 `payload.input` 等于 LLM response 中对应 `toolCall.input`
- [ ] 该 `tool.requested` 有恰好一个配对的 `tool.responded`，`responded.causedBy === requested.id`
- [ ] `tool.responded.payload.output` 不为 `undefined`（或者携带 `error`，二选一）
- [ ] 同一次决策的 `llm.requested` / `llm.responded` 共享同一个 `requestHash`
- [ ] Trajectory 中存在 `llm.call` span，其 `attributes.turn` 等于该决策所在的 turn 编号
- [ ] 整个 inspection 流程不触发任何额外 LLM 调用

## 不在此 story 范围

- **Timeline / filtering / 整体浏览** → 见 [s-002](./s-002-inspect-a-completed-run.md)
- **从最终 artifact 反推到 source（lineage forward）** → 见 [s-004](./s-004-lineage-from-artifact-to-source.md)
- **从 source 反推到所有引用它的 run（lineage reverse）** → 见 [s-014](./s-014-reverse-reference-lineage-query.md)
- **重放该决策点** → 见 [s-005](./s-005-deterministic-replay.md)
- **替换该决策后看 outcome 变化（fork）** → 见 [s-006](./s-006-fork-at-event-for-what-if.md)
