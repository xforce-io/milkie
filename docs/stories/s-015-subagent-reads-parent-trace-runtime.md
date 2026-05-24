---
id: s-015
title: Sub-agent reads parent's in-flight trace at runtime
status: draft
kind: scenario
subsystems:
  - agent-runtime
  - agent-trace
capability: runtime-trace-consumption
requires:
  - Sub-agent as named tool
  - Event-sourced Agent Trace event log
  - In-flight trace query API
owner: "@xupeng"
created: 2026-05-24
tests:
  - tests/e2e/s-015-subagent-reads-parent-trace-runtime.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-trace
  - ARCHITECTURE.md#representative-scenarios
  - ARCHITECTURE.md#cross-cutting-decisions-invariants
---

## 场景叙事

主 agent 跑到某个 decision point（例如生成了一份推荐 / 一个 claim），spawn 一个 verifier sub-agent 作为 named tool，将自己当前的 `runId` 传入。verifier 通过 CLI facade 查询 parent 的 **in-flight** trace：定位到 claim 对应的 LLM call、追溯到支撑它的 evidence events（retrieval 结果、tool outputs、引用的文档版本），**不重做 retrieval、不重新调用 LLM**，独立判断证据链是否成立，把判断作为 tool response 返回给 parent。

主 agent 据 verifier 的判断决定：继续输出 / 补充检索 / 标记 low-confidence / 切到 fallback 路径。整个过程作为一条 sub-agent invocation event 出现在 parent trace 里（按 invariant 11），verifier 自己的 trace 作为 nested sub-trace。

**这是 invariants 12-13 的标志性兑现**：trace 在 run 进行中即可被另一 agent 程序化消费，而不是 run 结束后的事后产物；消费接口是 CLI facade，verifier 拿到的不是定制 RPC，是同一套人也能调用的 CLI 命令。

与所有其他 story 的区别：其他 story 中 trace 消费都在 run 结束之后；这里 trace 必须在 in-flight 状态可读，且语义稳定（partial state、unfinished events 的查询契约要明确）。这是为什么 `In-flight trace query API` 是单独一个 requires 项。

传统 agent 框架做不了这一点：它们的 trace 是事后才完整的日志，消费接口面向 UI 而非 agent，没有"另一个 agent 中途读 trace"这个 first-class 操作。

> 待补：interaction flow / 验收准则 / 不在此 story 范围
