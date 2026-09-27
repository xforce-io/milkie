# 多轮会话、变量与外部投递

## 用户入口

- SDK：`getSessionHistory`、`getContextVar`、`setContextVar`、`deleteContextVar`、`listContextVars`、`attachProjection`、`listContextProjections`
- HTTP：`POST /session/history`、`POST /context/set`、`POST /context/get`、`POST /context/list`、`POST /projection/attach`、`POST /projection/list`

## 源码依据

- [src/runtime/Milkie.ts](../../../../src/runtime/Milkie.ts)
- [src/runtime/AgentRuntime.ts](../../../../src/runtime/AgentRuntime.ts)
- [src/context/assemble.ts](../../../../src/context/assemble.ts)
- [src/cli/serve.ts](../../../../src/cli/serve.ts)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 多轮/变量 | 同 context 两次 invoke，读取完整历史；变量 set/get/list/delete，验证两个 context 隔离 | 历史保持用户、助手和工具调用关系；变量按 context 隔离；缺失变量 HTTP 返回 null。 |
| 投递 | 附加 sourceRunId/displayText，随后发送短确认；再读取列表 | 投递在同一 user 消息中位于当前原话之前，原话处于末尾；限额和去重行为可观察。 |
| 空/错误 | 读取不存在会话；投递缺字段或 maxCount<1；SDK 设置 TTL 后过期读取 | HTTP 缺字段/非法限额为 400，会话不存在为 404；TTL 后变量不再可读。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/milkie-session-history.test.ts src/__tests__/MilkieContextVars.test.ts src/__tests__/MilkieContextProjections.test.ts src/__tests__/sessionHistory.test.ts src/__tests__/serve.test.ts
```

- [src/__tests__/milkie-session-history.test.ts](../../../../src/__tests__/milkie-session-history.test.ts)
- [src/__tests__/MilkieContextVars.test.ts](../../../../src/__tests__/MilkieContextVars.test.ts)
- [src/__tests__/MilkieContextProjections.test.ts](../../../../src/__tests__/MilkieContextProjections.test.ts)
- [src/__tests__/sessionHistory.test.ts](../../../../src/__tests__/sessionHistory.test.ts)
- [src/__tests__/serve.test.ts](../../../../src/__tests__/serve.test.ts)

## 已知缺口

HTTP 没有 deleteContextVar 路由，也没有 set 的 TTL 入参；不要从 SDK 能力推断 HTTP 对等。

## 对应 Stories

- [docs/stories/s-009-multi-turn-with-tool-error-recovery.md](../../../../docs/stories/s-009-multi-turn-with-tool-error-recovery.md)
