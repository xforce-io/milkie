---
id: s-011
title: Multi-state FSM with intent routing, slot filling, and escalation
status: active
kind: scenario
subsystems:
  - agent-runtime
  - agent-trace
capability: multi-state-fsm
requires:
  - FSM Core
  - working context
  - Action state with ctx.emit
  - Sub-agent as named tool
  - Trajectory observability
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-011-multi-state-fsm-intent-routing-and-slot-filling.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-runtime
  - docs/superpowers/specs/2026-05-16-agent-e2e-scenarios.md
---

> **#175 迁移说明（de-core multi-state FSM）**：core 已删除多态业务 FSM
> （`on:` 业务转移、`ctx.emit` 硬转移、`fsm.transition` 事件、#60/#31、
> sub-agent action 态）。下文叙述的多态拓扑（intent routing / 路由 / 升级）
> 不再是 core 能力，已降级为 userland 组合关注点。其 e2e
> （`tests/e2e/s-011-...e2e.test.ts`）已迁移为「单态 `type: llm` + slot-filling」
> 的轻量分档表达（设计 §6），并在文件头记录迁移依据。完整的
> DialogFlow → slot-filling + action-precondition POC 见
> `examples/repair-ticketing/`（设计 §10 切片 5）。详见
> `docs/design/175-decore-multistate-fsm.md`。

## 场景叙事

客服 Agent 用**自定义多状态 FSM**（`type: llm` 状态 + `type: action`
状态混合）处理一类完整的业务流程：意图识别 → 分流路由 → 槽填充（多
轮）→ 用户确认 → 执行。低置信度可直接升级人工。

工具 handler 通过 `ctx.emit()` 触发硬转移；置信度检查、槽位完整性检
查这些**确定性逻辑放在 handler 内**（不依赖 LLM 判断），保证 FSM 流
转可预测。

三条测试路径覆盖典型业务分支：
- **Path A 主路径**：意图清晰 → 收集 orderId/reason/preferRefund 三槽位 →
  用户确认 → spawn `cancellation-executor` 执行
- **Path B 澄清路径**：第一次意图模糊 → 进入 `clarifying` 追问 →
  第二次意图明确（billing）→ 路由 `billing-specialist` 处理
- **Path C 升级路径**：置信度 < 0.75 → 不进入槽填充，直接转移到
  `escalated`（terminal `type: llm` 状态生成转接说明）

子场景：在 confirming 状态用户拒绝（`USER_REJECTED`），FSM 退回
`collecting_slots`——这是 FSM 不只能向前的关键证据。

## 关键交互流

```
FSM 拓扑：
  intent_classification ──(INTENT_CANCELLATION)─→ collecting_slots
                       ──(INTENT_BILLING)──────→ routing_to_specialist
                       ──(INTENT_UNCLEAR)─────→ clarifying
                       ──(ESCALATE)───────────→ escalated (terminal)
  clarifying ──(INTENT_*)──→ ...
  collecting_slots ──(SLOTS_COMPLETE)──→ confirming
  confirming ──(USER_CONFIRMED)──→ executing
            ──(USER_REJECTED)──→ collecting_slots   ← 退回
  executing ──(DONE)──→ completed (terminal)
           ──(ERROR)──→ escalated

[Path A 主路径]
'我想取消订单'         → classify_intent(cancellation, 0.92) → INTENT_CANCELLATION
'订单号 ORD-456'       → collect_slot(orderId, 'ORD-456')
'收到货损坏了'         → collect_slot(reason, 'damaged')
'要退款'               → collect_slot(preferRefund, true) → SLOTS_COMPLETE
'确认'                 → confirm_action(true) → USER_CONFIRMED
                       → spawn cancellation-executor → DONE → completed

[Path B 澄清路径]
'我的账有问题'         → classify_intent(unclear, 0.55) → INTENT_UNCLEAR
'上个月扣了两次钱'     → classify_intent(billing, 0.91) → INTENT_BILLING
                       → spawn billing-specialist → DONE → completed

[Path C 升级路径]
'我要投诉你们！'       → classify_intent(unclear, 0.48) → confidence < 0.75
                       → ESCALATE → escalated → 生成转接说明
```

## 验收准则

**Path A:**
- [ ] FSM 状态序列依次出现 `collecting_slots` → `confirming` → `executing` → `completed`
- [ ] `classify_intent` 的 confidence ≥ 0.75
- [ ] `collect_slot` 被调用 3 次，分别填 orderId / reason / preferRefund
- [ ] working_memory 在过程中逐步累积槽位（中间 checkpoint 可见部分填充）
- [ ] `cancellation-executor` sub-agent 被 spawn 且 TaskResult.success
- [ ] 子场景：USER_REJECTED 后 FSM 退回 `collecting_slots`，
      `collecting_slots` 在 trajectory 中出现 2 次

**Path B:**
- [ ] FSM 状态序列含 `clarifying` 与 `routing_to_specialist` 与 `completed`
- [ ] `classify_intent` 被调用 2 次，第一次 `unclear`、第二次 `billing`
- [ ] `billing-specialist` sub-agent 被 spawn
- [ ] 两次 invoke 的 goal 字符相等

**Path C:**
- [ ] FSM 状态序列**仅**含 `escalated`（不进入 collecting_slots）
- [ ] `classify_intent` 的 confidence < 0.75
- [ ] 最终 output 含"人工 / 客服 / 转接"关键词
- [ ] `collect_slot` 调用次数为 0

## Path D：层级化槽填充（hierarchical slot filling）

`repair-ticketing` 示例在 `collecting_slots` 状态内复用层级实体解析器（HER core，
#167 / PR #170），把"站点 → 楼宇 → 部门 → 负责人"四级实体逐级解析进工作记忆——
**FSM 拓扑保持不变**，只是该状态挂载的工具换成 `lookup_entity` / `commit_entity`
这一对适配器（`examples/repair-ticketing/src/tools/entity-resolver.ts`，
`makeEntityResolverTools(resolver, requiredSlots)`）。解析器在启动时
`EntityResolver.load(schema, csv)` 构造一次，handler 全程复用、绝不重新解析，
**进程内调用，无子进程 / CLI 派生**。

**信任边界（关键）**：LLM 不能通过工具参数提供原始话术或上级约束。
- 话术（utterance）由适配器从 `ctx.currentTurn` 读取（运行时注入、整轮稳定，#164）；
  `lookup_entity` 的入参 schema **不含** utterance / query 字段。
- `pinned`（已确认的上级实体）由适配器从 `ctx.workingMemory` 派生；两个工具的
  入参 schema 均 **不含** `pinned` 字段。
- LLM 只能提供：目标 `level`、可选 `sessionHint`，以及（提交时）`selected`。

```
[Path D 层级槽填充]   FSM 拓扑不变；collecting_slots 内逐级 lookup→commit
'总部'        → lookup_entity({context:{level:'site'}})       → commit_entity('S01')  → WM.site=S01
'主楼'        → lookup_entity({context:{level:'building'}})   → commit_entity('B01')  → WM.building=B01
'IT网络部'    → lookup_entity({context:{level:'department'}}) → commit_entity('D03')  → WM.department=D03
'王芳'        → lookup_entity({context:{level:'assignee'}})   → commit_entity('E008') → WM.assignee=E008
                                                              → 四级齐备 → SLOTS_COMPLETE → confirming
```

交互细节：
- `lookup_entity`：适配器读 `ctx.currentTurn` 作为话术、从 WM 派生 `pinned` 过滤分支，
  调用 `resolver.lookup(...)`，把 `{ candidates, options, suggested }` 原样返回给 LLM；
  **不写 WM、不 emit**。
- `commit_entity`：LLM 用上一次 lookup 的 `options` / `suggested` 中的值作为 `selected`。
  适配器调用 `resolver.commit(...)` 并按状态路由：
  - `complete`：`WM.set(level, resolved.id)`；返回 `{ resolved }`。
  - `corrected`（`selected` 与 pinned 上级冲突）：取 pinned 对应的同名兄弟实体，
    `WM.set(level, resolved.id)` 写入修正后的 id；返回 `{ resolved, corrected }`，
    其中 `corrected: Record<string, string>`（被修正的层级名 → 修正后的 id；
    可能是提交层级以外的上级层级，故为映射而非单个字符串，对应
    `CommitOutput.correctedLevels`）。
  - `invalid_selection` / `missing` / `ambiguous` / `unknown`：不写 WM、不 emit，
    返回 `{ validationError }`。
- 当 `requiredSlots` 的每一级在 WM 中都已就位，适配器 `ctx.emit('SLOTS_COMPLETE')`，
  FSM 推进到 `confirming`。
- FSM 状态名（`collecting_slots`、`confirming` …）与任何层级名
  （site / building / department / assignee）都不相同——层级是 WM 里的数据，不是 FSM 状态。

**Path D 验收准则**（覆盖于
`examples/repair-ticketing/src/__tests__/entity-resolver.e2e.test.ts`）：
- [ ] 适配器进程内调用 `EntityResolver`，无子进程 / CLI 派生
- [ ] 恰好暴露两个 `ToolDefinition`：`lookup_entity`、`commit_entity`
- [ ] e2e 使用 stub gateway；FSM 状态名与任何层级名都不相等
- [ ] `lookup_entity` 返回 `{ candidates, options, suggested }`
- [ ] `commit_entity` 的 `resolved.id` 必出现在上一次 `lookup_entity` 的 `options` / `suggested` 中
- [ ] 提交成功（`complete` / `corrected`）写入 `WM.set(level, resolved.id)`
- [ ] 所有层级就位后 emit `SLOTS_COMPLETE`
- [ ] `invalid_selection` / `missing` / `ambiguous` / `unknown` → 不写 WM、不 emit、返回 `validationError`
- [ ] `corrected` 路径写入修正后的 id 并返回 `{ resolved, corrected }`，
  其中 `corrected` 类型为 `Record<string, string>`（非单个字符串）
- [ ] **话术隔离**：`lookup_entity` 入参不含 utterance/query；适配器从 `ctx.currentTurn` 完成检索
- [ ] **pinned 隔离**：两个工具入参均不暴露 `pinned`，仅由 WM 派生

## 不在此 story 范围

- **意图分类模型 / prompt 调优** → 不属于框架职责
- **escalated 后的人工对话流程**（已转人工，离开 milkie 范围）
- **Skill 加载 / 版本切换**（FSM 内 skill 行为）→ s-010
