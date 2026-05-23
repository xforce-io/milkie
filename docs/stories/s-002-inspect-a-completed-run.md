---
id: s-002
title: Inspect a completed agent run
status: draft
kind: scenario
subsystems:
  - agent-trace
capability: observability
requires:
  - Trajectory observability
  - Event-sourced Agent Trace event log
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-002-inspect-a-completed-run.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-trace
---

## 场景叙事

开发者跑完一个 agent run（成功或失败），拿到 runId 之后，想**前向浏览**这次运行里发生的一切——FSM 状态转移、LLM 请求/响应、tool 调用、working context 变化、错误事件——按时间线呈现，并能按事件类型/时间窗过滤。

不重跑、不分叉、不沿因果链追溯，**纯阅读**。这是大多数用户第一次接触 Agent Trace 的入口。

> 待补：interaction flow / 验收准则 / 不在此 story 范围
