# 证据查询与任务结果

## 用户入口

- 模型工具：`cite`、`declare_relation`、`get_run_io`、`get_execution`、`get_lineage`
- SDK：`recordTaskOutcome`、`getTaskOutcome`、`finalizeTaskOutcome`、`getFinalTaskOutcome`

## 源码依据

- [src/tools/lineage.ts](../../../../src/tools/lineage.ts)
- [src/tools/trace.ts](../../../../src/tools/trace.ts)
- [src/trace/diagnostics/buildLineageProjection.ts](../../../../src/trace/diagnostics/buildLineageProjection.ts)
- [src/runtime/Milkie.ts](../../../../src/runtime/Milkie.ts)
- [src/outcome/validateEvidence.ts](../../../../src/outcome/validateEvidence.ts)
- [src/outcome/TaskOutcomeFinalizationStore.ts](../../../../src/outcome/TaskOutcomeFinalizationStore.ts)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 证据 | 工具显式登记对象与关系；按 claimId/objectId 查询；从投递 run 读取上下文 | 查询返回登记关系及可定位证据，不从自然语言猜引用；selfOnly 拒绝或忽略未投递的外部 runId。 |
| 任务结果 | 执行完成后记录并读取 outcome，再按有效证据 finalize 并读取最终记录 | 可变 observation 与不可变 finalization 分开；重复同内容幂等，冲突不能覆盖已定结果。 |
| 失败/隔离 | 使用不存在 run、无效证据、不可持久化存储、跨会话 runId | 按对应契约拒绝；执行 completed 不会自动写 task outcome；测试具体错误码和数据无泄漏。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/lineageTools.test.ts src/__tests__/traceTools.lineage.test.ts src/__tests__/traceTools.selfOnly.test.ts src/__tests__/traceTools.deliveredDeref.test.ts src/__tests__/TaskOutcome.test.ts src/__tests__/TaskOutcomeFinalization.test.ts
```

- [src/__tests__/lineageTools.test.ts](../../../../src/__tests__/lineageTools.test.ts)
- [src/__tests__/traceTools.lineage.test.ts](../../../../src/__tests__/traceTools.lineage.test.ts)
- [src/__tests__/traceTools.selfOnly.test.ts](../../../../src/__tests__/traceTools.selfOnly.test.ts)
- [src/__tests__/traceTools.deliveredDeref.test.ts](../../../../src/__tests__/traceTools.deliveredDeref.test.ts)
- [src/__tests__/TaskOutcome.test.ts](../../../../src/__tests__/TaskOutcome.test.ts)
- [src/__tests__/TaskOutcomeFinalization.test.ts](../../../../src/__tests__/TaskOutcomeFinalization.test.ts)

## 已知缺口

s-004/s-014 的部分能力已有代码，不能照抄旧 blocked；s-015 完整“子执行读取父执行进行中数据”仍需专门证明，selfOnly 工具存在并不自动满足它。

## 对应 Stories

- [docs/stories/s-004-lineage-from-artifact-to-source.md](../../../../docs/stories/s-004-lineage-from-artifact-to-source.md)
- [docs/stories/s-014-reverse-reference-lineage-query.md](../../../../docs/stories/s-014-reverse-reference-lineage-query.md)
- [docs/stories/s-015-subagent-reads-parent-trace-runtime.md](../../../../docs/stories/s-015-subagent-reads-parent-trace-runtime.md)
- [docs/stories/s-016-record-and-query-task-outcome.md](../../../../docs/stories/s-016-record-and-query-task-outcome.md)
- [docs/stories/s-017-immutable-task-outcome-finalization.md](../../../../docs/stories/s-017-immutable-task-outcome-finalization.md)
