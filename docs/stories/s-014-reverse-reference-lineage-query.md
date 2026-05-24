---
id: s-014
title: Reverse-reference lineage query
status: draft
kind: scenario
subsystems:
  - agent-trace
capability: lineage-reverse-reference
requires:
  - Event-sourced Agent Trace event log
  - Lineage-by-typed-relations
owner: "@xupeng"
created: 2026-05-24
tests:
  - tests/e2e/s-014-reverse-reference-lineage-query.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-trace
  - ARCHITECTURE.md#representative-scenarios
  - docs/stories/s-004-lineage-from-artifact-to-source.md
---

## 场景叙事

调用方给定一个 source identifier（content hash、外部文档版本号、知识库实体 id、某段时间窗内某个 tool 的输出 hash），询问"过去 N 天哪些 run 的 lineage 引用了它"。系统反向遍历 lineage graph，返回结构化列表：(run_id, event_ids, customer/surface metadata)。

典型触发：
- `policy.md@v1.3` 被下架 → 谁需要重新回邮件、谁需要重发推荐
- 某个 retrieval 源被发现污染 → 哪些输出受影响、影响时间窗
- 某个 tool 在 5 月 12 日 misbehaved → 哪些 run 用了那段时间的 tool 输出

与 s-004 的区别：s-004 是 artifact → source（forward，单点起源追溯）；s-014 是 source → all dependents（reverse，影响面查询）。两者对应 lineage graph 的两个方向，是不同 query 形态，应当各自一个 contract。共同前提：lineage 必须是 typed graph，不是纯日志——否则反向查询根本表达不出来。

这是 invariant 12 "agent-first" 的另一个标志场景：传统 trace 是树状文本，agent 没法做反向遍历；milkie 的 typed event graph 让 reverse query 成为 first-class 查询。

> 待补：interaction flow / 验收准则 / 不在此 story 范围
