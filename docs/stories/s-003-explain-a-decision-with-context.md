---
id: s-003
title: Explain an agent decision with its full context
status: draft
kind: scenario
subsystems:
  - agent-trace
capability: explainability
requires:
  - Trajectory observability
  - Event-sourced Agent Trace event log
  - Working context snapshot at decision point
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-003-explain-a-decision-with-context.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-trace
---

## 场景叙事

开发者在浏览一个 run 时盯住了某一步（"它为什么调了 tool X / 转移到 state Y / 输出了这句话"），想**展开那一瞬间的完整推理材料**：agent 当时持有的 working context、发给 LLM 的 prompt、LLM 返回的响应、FSM 当前状态、可调用的能力清单。

回答的是 "why did the agent decide this"，而不是 "where did this artifact come from"——前者是单点深挖、材料化推理上下文；后者是反向 lineage（见 s-004）。

> 待补：interaction flow / 验收准则 / 不在此 story 范围
