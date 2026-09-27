# 会话导入与导出

## 用户入口

- SDK：`exportSession`、`importSession`；PortableSession.schemaVersion、expectedLatestRunId
- HTTP：`POST /session/export`、`POST /session/import`

## 源码依据

- [src/runtime/Milkie.ts](../../../../src/runtime/Milkie.ts)
- [src/runtime/PortableSession.ts](../../../../src/runtime/PortableSession.ts)
- [src/cli/serve.ts](../../../../src/cli/serve.ts)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 往返 | 产生多轮会话、变量和子执行，export 后导入新实例，再继续 invoke | 会话历史、变量、父子事件和恢复路由保留；源实例数据不被改写。 |
| 并发覆盖 | 带 expectedLatestRunId 导入，与现有较新状态冲突 | 拒绝覆盖；HTTP 为 409，成功时 conditionApplied 表明条件确实执行。 |
| 空/版本 | 导出未知 context；导入未知 schemaVersion 或未配置事件存储的实例 | 明确失败；HTTP 未知会话为 404，版本不支持为 400。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/portable-session.test.ts src/__tests__/serve.test.ts
```

- [src/__tests__/portable-session.test.ts](../../../../src/__tests__/portable-session.test.ts)
- [src/__tests__/serve.test.ts](../../../../src/__tests__/serve.test.ts)

## 已知缺口

会话包可能包含输入和工具原始数据，证据仅用人工测试内容。不能用导入后 invoke 成功替代 #259 的 checkpointId 验证。
