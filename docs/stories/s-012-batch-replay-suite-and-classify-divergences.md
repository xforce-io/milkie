---
id: s-012
title: Batch replay a saved suite and classify divergences
status: draft
kind: scenario
subsystems:
  - agent-trace
capability: suite-replay-and-diff
requires:
  - Event-sourced Agent Trace event log
  - Content-addressed response cache
  - Replay engine
  - Structural diff
  - Suite definition + batch replay
owner: "@xupeng"
created: 2026-05-24
tests:
  - tests/e2e/s-012-batch-replay-suite-and-classify-divergences.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-trace
  - ARCHITECTURE.md#representative-scenarios
---

## 场景叙事

调用方维护一个 saved suite（例如 `golden_v1`），由 N 条来自生产的真实 run 构成。准备 merge 一个改动（新 prompt、新 router、新 skill 版本），先把整个 suite 在新代码分支上 replay，得到每条 run 的 divergence 结果——byte-equivalent / structurally diverged。

对每条 diverged run，系统输出 structural diff（哪个 event 起开始分叉、分叉的因果链结构）。调用方（通常是 meta-agent，也可以是 dev）据此分类：regression / improvement / neutral。**回归测试集是真实流量切片，不是 synthetic eval；divergence 一定来自代码改动，因为 replay 是确定性的。**

与 s-005 的区别：s-005 是单 run 等价契约；s-012 是 N-run 批量 + diff 聚合的契约，重点在批量操作、结构化输出可被另一 agent 程序化消费分类。是 invariant 12 "agent-first" 的关键变现场景。

> 待补：interaction flow / 验收准则 / 不在此 story 范围
