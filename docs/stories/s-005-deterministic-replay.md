---
id: s-005
title: Deterministically replay a recorded agent run
status: draft
kind: scenario
subsystems:
  - agent-trace
  - agent-runtime
capability: replay
requires:
  - IOPort
  - Event-sourced Agent Trace event log
  - Content-addressed response cache
  - Non-determinism log
  - Replay engine
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-005-deterministic-replay.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-trace
---

## 场景叙事

用户拿一份完整的 agent run trace（event log + response cache + non-determinism log），让 milkie 重新跑一遍，**得到与原 run 完全相同的 state**。没有任何 LLM 实调用、没有新的随机性、没有时间差异——所有非确定性出口都被 trace 里记录的值替代。

典型用法：本地复现一个生产故障；examples / demo 在无 API key 环境下能 ship；regression test 把昨日的 run 当今日的 baseline。

> 待补：interaction flow / 验收准则 / 不在此 story 范围
