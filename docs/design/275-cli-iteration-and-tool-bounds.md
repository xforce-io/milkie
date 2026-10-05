# 原生 CLI 强制模型迭代预算并完整交付合法工具数据

- Issue：[#275](https://github.com/xforce-io/milkie/issues/275)
- L1：复用 Issue 中已经确定的产品决定，不另写产品稿。
- L2：本文件。验收仍以 Issue S1–S3 为准。此文档不是验证通过证据。

## 1 契约

`maxModelIterations` 只约束一次 `start`，不跨续接累计。整数范围 1 到 10000；50 合法。缺省时不安装迭代上限。非法值，以及 API 传输上的该字段，在任何模型请求之前返回 `unsupported_constraint`。

Grok 把上限传给 `--max-turns`。停止原因为 `max_turns`、`error_max_turns` 或事件 `max_turns_reached` 时，执行码为 `iteration_budget_exhausted`，并保留已关联的原生会话。Grok 1.0.46 会在 `max_turns_reached` 之后再输出 `end(cancelled)`；后到的取消、错误事件不能擦掉已经观察到的预算耗尽。没有该事件时，普通取消、超时和会话不匹配仍按各自原因分类，也不因设置了预算或发生过工具调用而推断耗尽。Pi 不能靠钩子抛错取消请求：扩展在 `before_provider_request` 中计数，超额时先写标记再 `ctx.abort()`，使请求信号在 HTTP 开始前已中止。设置预算时使用原生提供方包装把 `maxRetries` 设为零，关闭不经过计数钩子的内部重试；Pi 上层重试仍经过该钩子。通过 `session_before_compact` 取消 Pi 的自动及手动压缩，因为压缩请求不经过该计数钩子；过长会话可能返回上下文容量错误，而不能发出未计数的压缩请求。监督进程读到该标记后使用同一执行码，不把这次停止记成普通 `process_failed`。执行记录带 `iterationBudget: { limit, exhausted }`。状态为 `failed` 且本地已停止。同一上下文的下一次 `start` 继续使用原会话。

超时和工具调用次数不能代替模型迭代。

## 2 工具边界

三个上限分开计算，单位都是 UTF-8 字节：

| 对象 | 上限 |
|---|---|
| 输入里的单个原始字符串 | 256 KiB |
| 宿主工具成功结果，以及核对结果 | 256 KiB |
| 一行编码后的工具请求 | 2 MiB |

256 KiB 的 NUL 编码后仍低于 2 MiB，因此合法。恰好等于上限的结果原样交付，不截断，也不改成 uncertain 或触发第二次调用。超出上限的结果在处理函数已执行一次后记为 `rejected`，文案为 `Tool result exceeds 262144 bytes.`。超出上限的原始字符串或编码请求在调用处理函数之前记为 `invalid_input`。无法在 2 MiB 加 64 KiB 内形成完整请求行时关闭连接，不调用处理函数。

Grok 与 Pi 共用这条宿主通道。MCP 帧上限只放宽到能送进上述请求行，不单独放大业务结果。
