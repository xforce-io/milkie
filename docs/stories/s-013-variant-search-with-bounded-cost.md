---
id: s-013
title: Variant search with bounded amortized cost
status: draft
kind: scenario
subsystems:
  - agent-trace
  - evolution
capability: variant-search
requires:
  - Event-sourced Agent Trace event log
  - Content-addressed response cache
  - Fork primitive
  - Structural diff
owner: "@xupeng"
created: 2026-05-24
tests:
  - tests/e2e/s-013-variant-search-with-bounded-cost.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-trace
  - ARCHITECTURE.md#evolution
  - ARCHITECTURE.md#representative-scenarios
---

## 场景叙事

调用方给定 baseline run 和 N 个变体配置——每个变体只改动 run 中段往后的某个参数（synthesis prompt、特定 tool 的入参、某个 routing 阈值）。系统对每个变体在共同分叉点 fork、从分叉点起以新配置继续执行，前缀全部从 response cache 服务、不重付 LLM 钱。

返回 N 个新 run + N 份与 baseline 的 structural diff。**契约里包含成本边界**：实际 LLM 调用数 ≈ N × tail_size，前缀 cache 命中率可断言（test 会 mock LLM gateway 并 assert 实际 call count 不超过预期上界）。

与 s-006 的区别：s-006 是单次 fork 的功能契约；s-013 是 N 次 fork 的成本契约，是 Phase 3 content-addressed cache 的真正变现验证——cache 不只是 "replay 时复用"，更是 "变体探索几乎免费"，这一点不通过 N-run 测试就没人能信。

适用场景：自动 prompt 优化、Evolution 的 Outcome Collector 在 promote 前的回测、dev 在 review 阶段做 "as-if" 验证。前提：变体作用于 run 中段往后；若变体改 system prompt 则前缀失效，成本和传统框架一样，不在本 story 适用范围。

> 待补：interaction flow / 验收准则 / 不在此 story 范围
