# 结构化模型错误与关联服务日志（#202）

> 最后更新：2026-07-10  
> Issue: https://github.com/xforce-io/milkie/issues/202  
> 分支：`feat/202-structured-model-errors`

## 目标

模型网关失败不能再退化为无法诊断的 `Connection error.`。错误需要从
OpenAI-compatible adapter 经 runtime、event store 和 `serve` SSE 边界保留稳定、
可机读且不泄露敏感信息的语义，并由服务日志通过 `runId`、`contextId` 关联。

## 契约

新增 `ModelGatewayError`，其安全序列化结果包含：

- `code`：`MODEL_CONNECTION_ERROR`、`MODEL_TIMEOUT`、`MODEL_RATE_LIMITED`、
  `MODEL_AUTH_ERROR`、`MODEL_BAD_RESPONSE` 或 `MODEL_UNKNOWN_ERROR`。
- `message`：兼容现有消费者的安全文本。
- `phase`：`request`、`stream_open`、`stream_read` 或 `response_parse`。
- `provider`、`model`、可选 HTTP `status`。
- `retryable`：由模型边界明确给出。

原始异常只作为进程内 `cause` 保存。API key、认证头、请求头、prompt 和原始响应体
不得进入事件或 SSE。

SSE `error` 与终止 `agent.run.completed` 帧新增 `error` 对象，同时保留顶层
`message`/字符串输出兼容旧客户端。

## 实现边界

1. `OpenAICompatibleAdapter.complete()` 和 `.stream()` 统一归一化 SDK/HTTP 异常。
2. runtime 的失败事件保存同一安全 envelope。
3. `serve.streamTurn` 记录一条带 `runId`、`contextId` 和 envelope 字段的 error 日志。
4. Milkie 不实现业务重试、provider failover 或 circuit breaker。

## 测试计划

- **单元测试**：错误映射表、阶段分类、retryable、cause 和敏感信息脱敏。
- **集成测试**：假网关错误穿过 AgentRuntime 后，JSONL 终止事件保留安全 envelope。
- **功能测试**：真实 HTTP/SSE server 返回 `error` 与终止帧，并只写一条关联日志。
- **端到端测试**：真实 `milkie serve` 子进程连接确定性失败的 OpenAI-compatible stub，
  验证 SSE、JSONL、trace report，以及失败后 `/health` 仍可用。
- **构建/清单验证**：TypeScript build、相关 lint、现有 deterministic suite。

## 验收标准

- Alfred 可根据 `error.retryable` 决策，不再依赖错误字符串。
- 操作者可由任一失败 run 定位 provider、model、phase 和错误类型。
- 旧 SSE 客户端仍可读取人类可读 message。
- 失败路径不包含任何 secret 或 prompt 内容。
