---
id: s-001
title: ReAct agent with intra-agent parallel tools (plan-and-act)
status: draft
kind: scenario
subsystems:
  - agent-runtime
  - agent-trace
capability: plan-and-act
requires:
  - FSM Core
  - working context
  - Direct LLM/tool execution
  - Trajectory observability
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-001-react-with-intra-agent-parallel-tools.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-runtime
  - docs/superpowers/specs/2026-05-16-agent-e2e-scenarios.md
---

## 场景叙事

一个开发者构建竞品分析 Agent：让 LLM 先用 cognitive toolbox 制定计划、再
并发执行多个搜索、最后写出报告。整个过程在**单一 Agent 内**完成，不涉及
sub-agent。

这个 story 校验 milkie 最基础也最重要的能力：FSM Runtime（react 配置）+
intra-agent 并行（单次 LLM 响应输出多个 `tool_use` block，Runtime 并发执
行后合并 observation）+ Cognitive toolbox 通过 `workingMemory` 维护中间
状态。

它对应 Agent Runtime / Execution / Agent Trace 三个子系统的协作：
- Agent Runtime 跑 react FSM
- Execution 提供 cognitive / search / filesystem 三个 toolbox 的算子
- Agent Trace 记录每次 LLM 调用、工具调用和 working context 变更，供断
  言回放

## 关键交互流

```
test → milkie.invoke({
  agentId: 'analyst',
  goal: '分析 Product A/B/C 的核心功能差异',
  input: '输出 Markdown 报告到 ./test-output/case1/report.md'
})

analyst (react FSM, single Agent)
  │
  ├─ Turn 1 — plan
  │    LLM 输出: 1 个 tool_use
  │      create_plan(steps=[搜索 A/B/C, 对比分析, 写报告])
  │    Runtime: 执行 create_plan
  │      → workingMemory.plan = { steps: [...all pending...] }
  │
  ├─ Turn 2 — act (intra-agent 并行)
  │    LLM 输出: 3 个 tool_use（同一次响应内）
  │      web_search('Product A ...')   ┐
  │      web_search('Product B ...')   ├─ allSettled 并发执行
  │      web_search('Product C ...')   ┘
  │    Runtime: 合并 observation，进入下一 turn
  │
  ├─ Turn 3 — 更新进度 + 写报告
  │    LLM 输出: 2 个 tool_use
  │      update_step(0, status='done')
  │      write_file('./test-output/case1/report.md', content=...)
  │
  └─ Turn 4 — 完成
       LLM 输出: 1 个 tool_use + text
         update_step(2, status='done')
         "分析报告已生成: ./test-output/case1/report.md"
       FSM 终止
```

模型固定为 `claude-haiku-4-5-20251001`，工具 fixture 化（`web_search`
按 query 返回预置内容，`write_file` 写到 `./test-output/case1/`）。

## 验收准则

- [ ] 调用 `milkie.invoke(...)` 返回成功，且产生了一个 agentRunId
- [ ] `create_plan` 在第一个 `web_search` 之前被调用（plan 先于 act）
- [ ] `create_plan` 写入 `workingMemory.plan`，初始所有步骤为 `pending`
- [ ] 一个 turn 内出现 3 个 `web_search` tool 调用，turn 编号相同
- [ ] 这 3 个 `web_search` 时间上有重叠（最晚开始 < 最早结束 → 确实并发）
- [ ] `update_step` 至少被调用 2 次且至少有 2 次 `status='done'`
- [ ] `write_file` 在文件系统中真实写入了 `./test-output/case1/report.md`
- [ ] 报告内容同时包含 `Product A`、`Product B`、`Product C`
- [ ] Trajectory 中无 `agent.spawn` span（确认单 Agent、无 sub-agent）
- [ ] 所有 `llm.call` span 的 provider 为 `anthropic`、model 含 `haiku`

## 不在此 story 范围

- **Inter-agent 并行 / sub-agent 编排** → 见未来的 sub-agent fan-out story
- **多轮对话 / context 跨 invoke 复用** → 见未来的 multi-turn-resume story
- **中断与恢复** → 见未来的 interrupt-resume story
- **Skill 渐进加载 / A/B 实验** → 见未来的 Evolution 相关 story
- **错误恢复 / tool retry** → 见未来的 error-handling story
