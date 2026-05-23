---
id: s-006
title: Fork a run at an event to explore a counterfactual
status: draft
kind: scenario
subsystems:
  - agent-trace
  - agent-runtime
capability: fork
requires:
  - IOPort
  - Event-sourced Agent Trace event log
  - Content-addressed response cache
  - Fork primitive
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-006-fork-at-event-for-what-if.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-trace
---

## 场景叙事

用户对一个 200 步的 run 提问"如果我在第 150 步换一个 prompt / 换一个工具 / 改一个配置，结果会怎样"。系统在第 150 个事件处分叉：**前 149 步从 cache 服务，不重付 LLM 钱**，从第 150 步起以新配置继续跑。父分支不受影响，可与子分支结构化 diff。

这是 Evolution 的 Outcome Collector 评估变体的核心机制，也是 Human reviewer 做 "what if I had…" 实验的廉价方式。

> 待补：interaction flow / 验收准则 / 不在此 story 范围
