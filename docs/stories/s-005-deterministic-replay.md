---
id: s-005
title: Deterministically replay a recorded agent run
status: active
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
  - docs/stories/s-006-fork-at-event-for-what-if.md
---

> Phase 3 provides structural replay; byte-identical pending Phase 4 non-determinism log.

## 场景叙事

用户拿一份完整的 agent run trace（event log + response cache + non-determinism log），让 milkie 重新跑一遍，**得到与原 run 完全相同的 state**。
（Phase 3 已实现结构等价：status / output / 关键事件序列一致；byte-identical 时间戳与 UUID 仍依赖 Phase 4 的 non-determinism log。）
没有任何 LLM 实调用、没有新的随机性、没有时间差异——所有非确定性出口都被 trace 里记录的值替代。

典型用法：本地复现一个生产故障；examples / demo 在无 API key 环境下能 ship；regression test 把昨日的 run 当今日的 baseline。

> 待补：interaction flow / 完整验收准则（既有 e2e 为权威实现锚点）。

## 不在此 story 范围

- **Fork / debug rerun**（从某事件分支并修改决策以修复**当前任务**）→ 见 [s-006](./s-006-fork-at-event-for-what-if.md)。**Replay ≠ Fork**：replay 忠实重放已录 log，不声称修好错误决策
- **工具瞬时 Retry** → 见 [s-009](./s-009-multi-turn-with-tool-error-recovery.md)
- **Task outcome 写入** → 见 [s-016](./s-016-record-and-query-task-outcome.md)
