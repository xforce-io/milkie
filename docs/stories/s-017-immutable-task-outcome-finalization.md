---
id: s-017
title: Finalize an immutable task outcome with evidence
status: active
kind: scenario
subsystems:
  - agent-runtime
  - agent-trace
capability: task-outcome-finalization
requires:
  - Agent Trace event log (basic)
  - Trace object store
  - Task outcome finalization store
owner: "@xupeng"
created: 2026-08-09
tests:
  - tests/e2e/s-017-immutable-task-outcome-finalization.e2e.test.ts
related:
  - https://github.com/xforce-io/milkie/issues/227
  - docs/design/227-task-outcome-finalization.md
  - docs/stories/s-016-record-and-query-task-outcome.md
  - ARCHITECTURE.md#outcome-task-outcome
---

## 场景叙事

一个 agent run 已结束，验证器已经得到可引用的 Trace event 与内容地址对象。验证器要把任务结果、验证主体声明和证据一起封账，使后续评测、审计或并发调用方看到同一个不可覆盖的 final result。

封账不是现有 outcome observation 的替代品。s-016 允许人或 eval 持续追加 `task.outcome.recorded` 并查询最新 observation；本 story 的 finalization 每个 run 只能成功一次。后续 observation 可以继续存在，但不能改变 final view。

验证器身份的认证和授权在 milkie 之外的 trusted boundary 完成。milkie 保存 `verifierClaim`，验证 evidence ref 的同 run 归属与 object content hash，但不自行判断业务是否成功，也不把 claim 宣称为已认证身份。

## 关键交互流

```
evidence path（crash-safe 模式）
  ├─ invoke / Recording Tool 产生已结束 run
  ├─ Trace 含且仅含一个 agent.run.completed
  ├─ event evidence 可按 eventId 引用，EventStore 确认 run file 耐久
  └─ object evidence：
       · canonical bytes 写入 TraceObjectStore → sha256 content hash
       · 同 run object.created 记录 objectId + 相同 hash
       · ObjectStore 确认 object inode 与目录项耐久

first finalization
  └─ finalizeTaskOutcome({
       runId,
       expectedState: 'unfinalized',
       finalizationId,
       value: 'success' | 'failure' | 'partial' | 'unknown',
       verifierClaim: { type: 'eval', id: '...' },
       evidence: [
         { kind: 'event', eventId },
         { kind: 'object', objectId, hash },
       ],
     })
       → { status: 'finalized', final }

query（File final 每次返回前确认 target 目录项耐久）
  └─ getFinalTaskOutcome(runId)
       → state === 'finalized'
       → value / verifierClaim / evidence / intentHash / recordHash 可读
       → 重建全部 store 实例后，run / object / final 仍可取得并复验

retry and conflict
  ├─ 同 finalizationId + 同 intent
  │    → { status: 'idempotent', final: sameExisting }
  ├─ 同 finalizationId + 不同 intent
  │    → { status: 'conflict', conflict: { kind: 'idempotency_key_reused' }, existing }
  └─ 不同 finalizationId（相同或相反 value）
       → { status: 'conflict', conflict: { kind: 'already_finalized' }, existing }
```

## 验收准则

### S1. 验证器用证据封账任务结果

- [ ] 真实 run 已结束，并产生可引用的同 run event evidence 与带 hash 的 object evidence
- [ ] object evidence 的 canonical bytes 可从 TraceObjectStore 读取且重算 hash 与 ref/Trace event 一致
- [ ] crash-safe 第一次合法 finalization 只有在 event/object/final 均耐久确认后返回 `status: 'finalized'`
- [ ] 按 runId 查询得到一个且仅一个 final record；state/value/verifierClaim/evidence 与提交一致
- [ ] 关闭并重建 Event/Object/Final store 实例后，run、object bytes 与 final 仍可取得，object/record hash 可按公开 canonical 契约重算
- [ ] 追加相反的 s-016 observation 后，observation view 更新而 final view 不变
- [ ] 封账与查询路径不调用真实 LLM provider

### S2. 冲突封账不会覆盖已有结果

- [ ] 不同 finalizationId 并发提交时恰一项 `finalized`，其余 `conflict.kind === 'already_finalized'`
- [ ] loser 返回的 existing 与 winner 的 final 具有同一 recordHash，持久化记录数始终为 1
- [ ] winner 以同 finalizationId + 同 intent 重试返回 `idempotent`，recordHash 和内容不变
- [ ] 同 finalizationId + 不同 intent 返回 `conflict.kind === 'idempotency_key_reused'`
- [ ] 相同 value 但不同 finalizationId 仍返回 `conflict.kind === 'already_finalized'`，不能伪装成重试
- [ ] link 后 directory fsync 失败时，winner/reader/loser 在自身确认成功前都不得收到 final/existing，只返回可重试的 commit_unknown
- [ ] malformed/tampered existing 以 corruption error fail closed，不创建替代 final

> 详细契约、耐久边界与测试分层以 [#227 L2 design](../design/227-task-outcome-finalization.md) 为事实源；目标 E2E 为 `tests/e2e/s-017-immutable-task-outcome-finalization.e2e.test.ts`。

## 不在此 story 范围

- **可多次追加、last-write-wins 的 outcome observation** → 见 [s-016](./s-016-record-and-query-task-outcome.md)
- **milkie 自动判断业务任务成败** → 非目标；value 由 trusted verifier 提交
- **验证器身份认证/授权** → 上层 trusted boundary 负责；milkie 只保存 claim
- **证据生成与领域解释** → Trace/ObjectStore 与领域验证器负责
- **reopen、override、delete final result** → 禁止
- **管理员级防篡改** → 普通 hash 不提供该保证；需签名/WORM/transparency 能力
- **LLM/Tool 截止时间与失败 Replay** → 见 #228/#229，独立契约
