---
id: s-009
title: Multi-turn conversation with tool error recovery
status: draft
kind: scenario
subsystems:
  - agent-runtime
  - agent-trace
capability: multi-turn-with-error-handling
requires:
  - FSM Core
  - working context
  - State stores
  - Error handling FSM transition
  - Trajectory observability
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-009-multi-turn-with-tool-error-recovery.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-runtime
  - docs/superpowers/specs/2026-05-16-agent-e2e-scenarios.md
---

## 场景叙事

一个 order-analyst agent 配置成"LLM 状态无 `on.DONE`"——即 LLM 输出
后等待下一条用户消息（多轮对话）。Goal 在多轮间保持不变，每轮的
`current_turn` 变化，`history` 跨 invoke 累积，`contextId` 复用作为
关联键。State store 用 Redis。

第 1 次 invoke 中，`query_orders` 工具首次调用模拟超时并标记
`retryable: true`，FSM 自动从 `analyze` 转移到 `error_handling`，重
试 `query_orders`，成功后再转回 `analyze`，LLM 给出初步分析。

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
  → error_handling state（FSM 自动转移）
  → 重试 query_orders → 成功
  → analyze state（FSM 转回）
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
- [ ] 第 2 次 invoke 的 `context.history` 包含第 1 轮的 `query_orders` 痕迹
- [ ] trajectory 里存在 `fsm.transition` 进入 `error_handling`
      与从 `error_handling` 转出
- [ ] `query_orders` 共被调用 2 次（首次超时、第二次成功）
- [ ] 第 2 次 invoke 的 output 含最终判断（"正常 / 异常 / 判断 / 结论"等关键词）

## 不在此 story 范围

- **中断与恢复**（Interrupt / Resume）→ s-008
- **不可重试错误的终止行为**（非 retryable） → 未来的 error story
- **多 contextId 之间的隔离** → 未来的 context-isolation story
