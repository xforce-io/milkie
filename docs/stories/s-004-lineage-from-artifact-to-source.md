---
id: s-004
title: Trace lineage from an artifact back to its source
status: draft
kind: scenario
subsystems:
  - agent-trace
capability: lineage
requires:
  - Event-sourced Agent Trace event log
  - Lineage query API
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-004-lineage-from-artifact-to-source.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-trace
---

## 场景叙事

agent 产出了一个 artifact（claim、object、relation、最终输出片段），用户问"这个东西是哪来的、依据是什么"。系统沿 lineage 反向追溯，给出**结构化的因果链**：artifact → 产生它的 behavior → 触发它的 event → 对应的 LLM call → 原始 prompt 与外部数据来源。

适用领域包括 due diligence、合规审计、医疗/法律证据链、研究可复现性——任何"理由和答案同等重要"的场景。这是 ActiveGraph 论文中 diligence pack 例子的能力镜像。

> 待补：interaction flow / 验收准则 / 不在此 story 范围
