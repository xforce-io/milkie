# 模型连接与单次调用

## 用户入口

- SDK：`complete`；`collectFromPrefix`、`resolveAndParseConnection`、`assembleApiGateway`、`createGateway`
- HTTP：`POST /llm`，非流式 JSON 和流式 SSE；AgentConfig.model/models、图像能力配置

## 源码依据

- [src/runtime/Milkie.ts](../../../../src/runtime/Milkie.ts)
- [src/connection/parse.ts](../../../../src/connection/parse.ts)
- [src/connection/assemble.ts](../../../../src/connection/assemble.ts)
- [src/gateway/GatewayFactory.ts](../../../../src/gateway/GatewayFactory.ts)
- [src/cli/serve.ts](../../../../src/cli/serve.ts)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 正常 | 用本地假模型服务配置两类协议，调用 complete 与 /llm；再选 tier 和 stream | 模型选择、temperature、文本和 usage 符合请求；流式以 done 结束。 |
| 图像 | 传入文本和图像消息，分别启用/禁用 imageInput | 支持时适配器保留图像；不支持时明确拒绝，不静默丢图。 |
| 错误/空 | 测试缺少连接字段、混用旧新字段、空 messages、模型 HTTP 错误 | 配置错误在请求前暴露；/llm 空 messages 为 400，流式错误有 error 和 done。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/modelConnectionContract.test.ts src/__tests__/milkie-complete.test.ts src/__tests__/gatewayImageMessage.test.ts src/__tests__/ModelGatewayError.test.ts src/__tests__/serve.test.ts
```

- [src/__tests__/modelConnectionContract.test.ts](../../../../src/__tests__/modelConnectionContract.test.ts)
- [src/__tests__/milkie-complete.test.ts](../../../../src/__tests__/milkie-complete.test.ts)
- [src/__tests__/gatewayImageMessage.test.ts](../../../../src/__tests__/gatewayImageMessage.test.ts)
- [src/__tests__/ModelGatewayError.test.ts](../../../../src/__tests__/ModelGatewayError.test.ts)
- [src/__tests__/serve.test.ts](../../../../src/__tests__/serve.test.ts)

## 已知缺口

连接契约接受 agent-cli 配置不代表本仓库已实现 agent-cli 执行器；assembleApiGateway 仅装配 API 连接。真实远端模型可用性须单独验证。
