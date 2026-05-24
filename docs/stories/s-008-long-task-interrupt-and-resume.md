---
id: s-008
title: Interrupt a long-running agent and resume from checkpoint
status: active
kind: scenario
subsystems:
  - agent-runtime
  - agent-trace
capability: interrupt-resume
requires:
  - FSM Core
  - State stores
  - Yield point + interrupt signal
  - Supervisor tree (interrupt propagation)
  - Trajectory observability
owner: "@xupeng"
created: 2026-05-23
tests:
  - tests/e2e/s-008-long-task-interrupt-and-resume.e2e.test.ts
related:
  - ARCHITECTURE.md#agent-runtime
  - docs/superpowers/specs/2026-05-16-agent-e2e-scenarios.md
---

## 场景叙事

一个 analyst agent 顺序处理 10 个 chunk（每个调一次 `process_chunk`
工具，工具是幂等的）。处理到第 3 个 chunk 时外部注入中断信号：Runtime
在最近的 yield point 检测到中断、保存 checkpoint 到 state store
（SQLite）、把 FSM 切到 `paused`、抛 `InterruptSignal`。

之后 `milkie.resume(checkpointId)` 从 checkpoint 恢复——LLM 看到历史
里前 3 个 chunk 已处理，从 chunk 4 继续，到 chunk 10 完成。整个过程
跨中断的 trajectory 是连续的，10 个 tool 调用各自独立 `toolCallId`，
chunk 1-3 不被重复执行。

子场景：父 agent 启动两个并发 sub-agent 时被中断，**中断信号传播到
supervisor tree**——父 agent 把中断推入两个 child 的 event queue，每
个 child 在自己的 yield point 保存 checkpoint 并返回 `interrupted`
的 TaskResult，父 agent 的 checkpoint 记录两个 child checkpoint id。

## 关键交互流

```
[主流：单 Agent 中断恢复]

test → milkie.invoke(analyst, goal='处理 dataset-42 的 10 个 chunk')

analyst (react FSM, max_iterations=20)
  ├─ turn 1: 制定计划
  ├─ turn 2: process_chunk(1) → ok
  ├─ turn 3: process_chunk(2) → ok
  └─ turn 4: process_chunk(3) → ok
              ↑ yield point
test ──→ eventQueue.push({ type: 'interrupt' })
              │
              ├─ yield point 检测到 interrupt
              ├─ saveCheckpoint() → SQLite checkpoint:latest
              ├─ FSM → paused
              └─ throw InterruptSignal

test → milkie.resume(checkpointId)
analyst (resumed)
  ├─ turn 5: process_chunk(4) → ok
  ├─ ...
  └─ turn 11: process_chunk(10) → ok, 输出汇总

[子流：Supervisor Tree 中断传播]

test → milkie.invoke(orchestrator)
orchestrator
  ├─ worker-a(task='A')  ┐ allSettled 中
  └─ worker-b(task='B')  ┘
test ──→ orchestrator.eventQueue.push({ type: 'interrupt' })
  │  ├─ orchestrator 把 interrupt 推给 worker-a / worker-b
  │  └─ 等 allSettled
  worker-a/b 各自 yield point → saveCheckpoint → TaskResult.interrupted
orchestrator → saveCheckpoint（children 记录两个 child checkpoint id）
            → FSM paused
```

## 验收准则

**主流：**
- [ ] 中断后 checkpoint 的 `fsm.currentState == 'paused'`
- [ ] 中断时 `sequence == 3`（已完成 3 个 chunk）
- [ ] checkpoint 的 `pendingEvents` 为空（中断前无积压事件）
- [ ] resume 后总 `process_chunk` 调用 10 次，10 个独立 `toolCallId`
- [ ] chunk 1-3 在中断前首次执行（`from_cache == false`），resume 后不被重复调用
- [ ] resume 后调用的 chunk ids 排序后等于 `[1..10]`，无重复
- [ ] 最终 trajectory `status == 'completed'`，跨中断 span 连续，无 gap

**子流（supervisor tree）：**
- [ ] 父 agent checkpoint 的 `children` 数组长度为 2
- [ ] 两个 child 的 status 均为 `interrupted`，均有非空 `checkpointId`

## 不在此 story 范围

- **错误恢复（tool retryable 错误）** → s-009
- **多轮对话上下文延续**（非中断意义上的延续）→ s-009
- **崩溃恢复 / 进程重启后从 checkpoint 拉起**（不在该场景内）→ 未来
