---
id: s-006
title: Fork a failed run at an event to repair the current task
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
  - ARCHITECTURE.md#retry-fork-and-reconfigure-action-vocabulary
  - docs/stories/s-005-deterministic-replay.md
  - docs/stories/s-009-multi-turn-with-tool-error-recovery.md
  - docs/stories/s-016-record-and-query-task-outcome.md
---

## 场景叙事

一次长 run 已经结束（或已判定任务失败）：执行路径里某一步决策错了——例如第 N 个事件处选了错误工具、用了错误 prompt 片段、或 FSM 走了错误分支。操作者要**修好当前这次任务**，而不是「再盲跑一遍」或「只忠实重放日志」。

系统在指定 **event** 处分叉（fork）：

- **分叉点之前**的共享前缀从 response cache / 已录事件服务，不重付 LLM 成本；
- **分叉点及之后**按修正（换工具选择、局部配置、或其它 fork 参数）继续执行，产生**新的 branch run**；
- **父 run 不变**；父子可后续做结构化 diff（完整 diff 见 s-012）。

主动机是 **当前任务修复（debug rerun）**。  
「what if 换一个配置会怎样」的反事实实验是同一机制的次要用法，也可服务后续变体评估，但本 story 验收以**修复路径**为准。

与相邻能力的边界（ARCHITECTURE 动作词汇）：

| 能力 | 本 story？ | 说明 |
|------|------------|------|
| **Retry** | 否 | 同一配置下瞬时失败重试 → [s-009](./s-009-multi-turn-with-tool-error-recovery.md) |
| **Replay** | 否 | 忠实重放已录 log，证明可复现，不声称修好错误决策 → [s-005](./s-005-deterministic-replay.md) |
| **Fork** | 是 | 从事件点分支并改决策/参数，救当前任务 |
| **Reconfigure** | 否 | 换 agent/skill 版本面向**未来**任务 → [s-010](./s-010-skill-versioned-load-and-ab-experiment.md) |

可选：fork 前可有 task outcome / 薄 Attribution 记录；**均非 fork 的前置条件**。

## 关键交互流

```
准备
  ├─ 已有 parentRunId（一次已录 run；事件数 ≥ N，N 为分叉点）
  ├─ （可选）task outcome = failure — 见 s-016；非必须
  └─ 选定 fork 点 eventId（或序号），及 fork 修正（例：替换该步工具输入 / 配置补丁）

fork
  ├─ milkie.fork({ runId: parentRunId, atEventId, patch? })
  │    → { branchRunId, parentRunId, forkEventId }
  ├─ 前缀：branch 不重新实调 LLM 覆盖已 cache 的共享段
  └─ 后缀：从 fork 点起按 patch 继续，写出 branch 自己的 event log

观察
  ├─ parent run 的事件序列与 fork 前一致（未改写）
  ├─ branchRunId ≠ parentRunId
  ├─ branch 从 fork 点起与 parent 可观测分歧（至少一处 LLM/tool/FSM 差异或最终 output 差异）
  └─ （可选）对 branch 写 task outcome；不自动 promote 配置
```

> API 形状以实现 design 为准；本 story 约束的是**按事件分叉、父不变、前缀便宜、后缀可改**。

## 验收准则

- [ ] 给定已完成的 `parentRunId` 与合法 `atEventId`（或等价位置），fork 返回新的 `branchRunId`
- [ ] `branchRunId !== parentRunId`
- [ ] fork 之后再次读取 parent 的 event log，与 fork 前一致（父分支不受影响）
- [ ] branch 的 event log 在 fork 点之前与 parent 结构对齐（共享前缀；具体等价定义：相同 requestHash 序列或实现规定的 prefix 契约）
- [ ] 共享前缀段**不产生新的真实 LLM 调用**（cache / 录制回放服务；stub 下 callCount 不增加前缀长度）
- [ ] 从 fork 点起，branch 应用了声明的修正（可观测：不同 tool 输入、不同配置字段、或不同后续事件）
- [ ] branch 能跑到终止状态（`completed` / `error` / `interrupted` 之一），并产生可查询的 trace
- [ ] （语义）同一 parent 上 Retry 路径（s-009）与本 fork API 不是同一入口；文档/测试不把「无 patch 整 run 重跑」称作 fork
- [ ] Attribution / task outcome **不是**调用 fork 的必填参数

## 不在此 story 范围

- **确定性 Replay（忠实重放，零修复语义）** → [s-005](./s-005-deterministic-replay.md)。Replay ≠ debug rerun ≠ Fork
- **工具 retryable / error_handling 瞬时 Retry** → [s-009](./s-009-multi-turn-with-tool-error-recovery.md)
- **Task outcome 的写入与查询** → [s-016](./s-016-record-and-query-task-outcome.md)（fork 可消费 outcome，但不实现 outcome）
- **批量 suite replay 与 divergence 分类** → [s-012](./s-012-batch-replay-suite-and-classify-divergences.md)
- **有界成本的多变体搜索** → [s-013](./s-013-variant-search-with-bounded-cost.md)
- **把 fork 结果自动 promote 为生产配置（Reconfigure / Evolution Gate）** → 非本 story
- **强制先写 Attribution 才能 fork** → 非目标；Attribution 可选
- **Lineage 正反向查询** → s-004 / s-014
