# #204 结构化运行时终态错误

最后更新：2026-07-12

## 问题

模型错误已有结构化 envelope，但 `MaxIterationsError` 只留下文本。调用方无法稳定地区分循环耗尽与模型或工具故障。

## 设计

扩展现有错误 envelope 的适用范围，不改变模型错误字段。`MaxIterationsError` 映射为：

- `code`: `MAX_ITERATIONS_EXCEEDED`
- `phase`: `agent_loop`
- `retryable`: `false`
- `message`: 保留当前可读文本

该 envelope 同时进入 `AgentResult`、持久化 `agent.run.completed` 和 serve SSE 的 `error`、`agent.run.completed` 帧。

## 测试计划

- 单元测试：异常映射和 completion payload。
- 集成测试：两轮上限的确定性循环返回结构化错误。
- 功能测试：`/chat` 两个终态 SSE 帧包含同一 envelope。
- 端到端测试：真实 sidecar 使用确定性循环模型耗尽次数，客户端和事件日志均取得相同错误字段。

## 兼容性

继续保留原错误文本；现有只读取字符串的消费者不受影响。
