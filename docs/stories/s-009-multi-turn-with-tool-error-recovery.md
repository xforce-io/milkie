---
id: s-009
title: Multi-turn conversation with tool error recovery
status: active
kind: scenario
subsystems:
  - agent-runtime
  - agent-trace
capability: multi-turn-with-error-handling
requires:
  - FSM Core
  - working context
  - State stores
  - Retryable tool error recovery
  - Trajectory observability
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-009-multi-turn-with-tool-error-recovery.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-runtime
  - ARCHITECTURE.md#retry-fork-and-reconfigure-action-vocabulary
  - docs/superpowers/specs/2026-05-16-agent-e2e-scenarios.md
  - docs/stories/s-006-fork-at-event-for-what-if.md
  - docs/stories/s-016-record-and-query-task-outcome.md
---

## 场景叙事

一个 order-analyst agent 配置成"LLM 状态无 `on.DONE`"——即 LLM 输出
后等待下一条用户消息（多轮对话）。Goal 在多轮间保持不变，每轮的
`current_turn` 变化，`history` 跨 invoke 累积，`contextId` 复用作为
关联键。State store 用 Redis。

第 1 次 invoke 中，`query_orders` 工具首次调用模拟超时并标记
`retryable: true`；Runtime 在同一 authored state 内做 lifecycle 重试
（非业务 `error_handling` transition span），成功后 LLM 给出初步分析。

第 2 次 invoke 用同一 `contextId`、同一 goal、新 input；agent 拿到
含第 1 轮对话的 history，结合新信息给出最终判断。

> **说明**：本 story 同时覆盖"多轮对话"与"tool 错误恢复"。两者在
> 该场景里耦合（同一 contextId 的连续两轮、第一轮里发生错误恢复），
> 暂作为一个故事保留；如果讨论后认为应拆分，按 README 的 "split
> into two stories" 规则处理。

## 关键交互流

```
[第 1 次 invoke]
milkie.invoke({
  agentId: 'order-analyst',
  goal: '分析订单 #12345 的异常原因',
  input: '订单金额超出阈值 3 倍',
})

  → analyze state
  → LLM: query_orders('12345') → 超时（retryable）
  → lifecycle 重试 query_orders → 成功（tool.call attempt 0 error / 1 ok）
  → LLM 输出初步分析 → DONE
  → 无 on.DONE → 等待下一条用户消息

[第 2 次 invoke]
milkie.invoke({
  ...,
  input: '客户历史消费记录显示为正常季节性采购',
  contextId: run1.contextId,
})

  → analyze state（history 含第 1 轮）
  → LLM 综合 history + 新 input → 输出最终判断
```

## 验收准则

- [ ] `goal` 在两次 invoke 的 checkpoint 中字符相等
- [ ] 两次 invoke 使用同一 `contextId`
- [ ] 第 2 次 invoke 的 checkpoint `context.regions` 中存在 `section === 'history'` 的 region，且其序列化内容包含第 1 轮用户输入稳定标识（如「订单 #12345 金额超出阈值 3 倍」）；tool 重试本身由独立 tool.call error/success 断言覆盖
- [ ] trajectory 里 `query_orders` 的 `tool.call` span ≥ 2，且至少 1 次 error + 1 次 success（优先校验 `attributes.attempt` 0/1）
- [ ] `query_orders` 共被调用 ≥ 2 次（首次超时、后续成功）
- [ ] 第 2 次 invoke 的 output 含最终判断（"正常 / 异常 / 判断 / 结论"等关键词）

## 不在此 story 范围

- **中断与恢复**（Interrupt / Resume）→ s-008
- **不可重试错误的终止行为**（非 retryable） → 未来的 error story
- **多 contextId 之间的隔离** → 未来的 context-isolation story
- **Fork / 当前任务决策修复** → 见 [s-006](./s-006-fork-at-event-for-what-if.md)。本 story 是 **Retry**（`retryable` 工具 + lifecycle 重试），不是 Fork
- **Task outcome（业务/任务成败）** → 见 [s-016](./s-016-record-and-query-task-outcome.md)；工具重试成功只影响 execution 路径，不自动等于 task success
