# 中断、恢复与执行边界

## 用户入口

- SDK：`interrupt`、`resume`、`getContextState`
- CLI：`agent interrupt <contextId>`、`agent resume <contextId>`
- HTTP：`POST /interrupt`、`POST /resume`、`POST /context/state`

## 源码依据

- [src/runtime/Milkie.ts](../../../../src/runtime/Milkie.ts)
- [src/runtime/AgentRuntime.ts](../../../../src/runtime/AgentRuntime.ts)
- [src/runtime/checkpointSchema.ts](../../../../src/runtime/checkpointSchema.ts)
- [src/trace/diagnostics/checkpointFromEvents.ts](../../../../src/trace/diagnostics/checkpointFromEvents.ts)
- [src/cli/serve.ts](../../../../src/cli/serve.ts)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 主路径 | 启动持续调用工具的执行，在活动中发送 interrupt，读取 context 状态，再恢复至停止 | 观察到中断终态和后续执行；工作状态保留；最终结果与执行事件对账。 |
| 精确恢复 | 预算耗尽后将返回 checkpointId 原样传给 resume；同 context 产生新快照后再恢复旧 ID | 应定位 ID 对应快照，而非最新快照；当前此项为已知失败 #259。 |
| 连续/错误 | 通过 context:<id>:checkpoint:latest 绕过 #259，连续恢复两次并触发正常停止和运行错误 | 每次应有可辨认起止、来源和终态；当前缺少起止记录 #260。 |
| 空 | 恢复未知 context 或 ID；查询未开始 context 状态 | 记录接口实际错误或空状态；不要凭缺少记录推断已完成。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/checkpoint-from-events.test.ts src/__tests__/checkpointSchema.test.ts src/__tests__/serve.test.ts src/__tests__/serve-persistence.test.ts
```

- [src/__tests__/checkpoint-from-events.test.ts](../../../../src/__tests__/checkpoint-from-events.test.ts)
- [src/__tests__/checkpointSchema.test.ts](../../../../src/__tests__/checkpointSchema.test.ts)
- [src/__tests__/serve.test.ts](../../../../src/__tests__/serve.test.ts)
- [src/__tests__/serve-persistence.test.ts](../../../../src/__tests__/serve-persistence.test.ts)

## 已知缺口

#259 与 #260 在 7e2687f 已复现。CLI/HTTP 使用 context 别名，不能用其成功证明 SDK 返回的 UUID 可用。SQLite 重建后恢复另见 14。

## 对应 Stories

- [docs/stories/s-008-long-task-interrupt-and-resume.md](../../../../docs/stories/s-008-long-task-interrupt-and-resume.md)
