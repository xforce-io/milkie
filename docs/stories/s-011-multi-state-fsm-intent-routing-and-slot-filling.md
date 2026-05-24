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

## 不在此 story 范围

- **意图分类模型 / prompt 调优** → 不属于框架职责
- **escalated 后的人工对话流程**（已转人工，离开 milkie 范围）
- **Skill 加载 / 版本切换**（FSM 内 skill 行为）→ s-010
