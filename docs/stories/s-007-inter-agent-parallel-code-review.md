---
id: s-007
title: Inter-agent parallel via named sub-agent tools
status: active
kind: scenario
subsystems:
  - agent-runtime
  - agent-trace
capability: inter-agent-parallel
requires:
  - FSM Core
  - Sub-agent as named tool
  - Direct LLM/tool execution
  - Trajectory observability
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-007-inter-agent-parallel-code-review.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-runtime
  - docs/superpowers/specs/2026-05-16-agent-e2e-scenarios.md
---

## 场景叙事

一个 orchestrator agent 在单次 LLM 响应里同时调用三个专项 reviewer
sub-agent（security / perf / style），每个 reviewer 是**独立 Agent 实
例**——独立 FSM、独立 working context、独立 trace。Runtime allSettled
join 三个 TaskResult，orchestrator 在下一 turn 把它们汇总成最终审查报告。

这是 milkie 的 **inter-agent 并行**：与 s-001 的 intra-agent 并行（一次
响应里多 tool_use）不同，这里 sub-agent 是命名工具，调用即启动独立实
例，适合子任务自身需要多步推理的场景。

## 关键交互流

```
test → milkie.invoke({
  agentId: 'review-orchestrator',
  goal: '审查 ./test/fixtures/code/target.ts',
  input: '请并行启动三个专项审查',
})

review-orchestrator (react FSM)
  │
  ├─ Turn 1 — fan-out
  │    LLM 输出: 3 个 tool_use（同一响应）
  │      security-reviewer({ file: '...target.ts' })  ┐
  │      perf-reviewer({ file: '...target.ts' })      ├─ allSettled
  │      style-checker({ file: '...target.ts' })      ┘
  │    Runtime: 每个 tool 调用 → 启动独立 sub-agent 实例
  │
  │  ┌─ security-reviewer (独立 FSM + ctx)
  │  │    └─ read_file(target.ts) → 找 SQL 注入 → TaskResult.success
  │  ├─ perf-reviewer       (独立 FSM + ctx)
  │  │    └─ read_file(target.ts) → 找 N+1 查询 → TaskResult.success
  │  └─ style-checker       (独立 FSM + ctx)
  │       └─ read_file(target.ts) → 找命名问题 → TaskResult.success
  │
  └─ Turn 2 — 汇总
       LLM 输出: text（三类发现汇总）→ react FSM 终止
```

固定 model `claude-haiku-4-5-20251001`；fixture 文件含三类已知问题。

## 验收准则

- [ ] trace 产生 3 个 `agent.spawn` span，子 agent 分别为
      `security-reviewer` / `perf-reviewer` / `style-checker`
- [ ] 3 个 sub-agent 的 `childTraceId` 互不相同（context 隔离）
- [ ] 3 个 `agent.spawn` 同属一个 turn（同一响应触发，inter-agent 并行）
- [ ] 3 个 sub-agent 的 `TaskResult` 状态均为 `success`
- [ ] orchestrator 最终 output 同时覆盖三类发现（safety / perf / style 关键词）
- [ ] `ResolvedManifest.subAgents` 精确记录三个 sub-agent 各自版本

## 不在此 story 范围

- **Intra-agent 并行**（同一 Agent 一次响应多 tool_use）→ s-001
- **Sub-agent 中断传播**（supervisor tree 中断）→ s-008
- **Sub-agent 之间的 trace 嵌套查看体验**→ 未来的 `sub-agent-trace-nesting`
