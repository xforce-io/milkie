# CLI 宿主工具

## 用户入口

SDK `ExecutionClient.start(contextId, input, { tools, forwarding }, handler)` 与 `ExecutionClient.toolCall(callId)`。`capabilities()` 增加 `hostTools`、`nativeCallId` 和 `forwarding`。无新增 HTTP、CLI 或页面。设计为 [product.md](../../../../docs/design/266-agent-cli-host-tools/product.md) v1，S1–S4 的子项归本功能。

## 源码依据

`src/execution/ExecutionClient.ts`、`worker.ts`、`hostTools.ts`、`mcp-server.ts`、`adapters.ts`、`types.ts`、`store.ts`。

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| S1.A1 | Grok：登记 `note` 并成功处理一次 | 调用成功且无 `nativeCallId`；记录关联 run 与 context |
| S1.A2 | Grok：处理函数拒绝 | 调用状态 `rejected` |
| S1.A3 | Pi：同 S1.A1 | 调用成功且有 `nativeCallId` |
| S1.A4 | Pi：同 S1.A2 | 调用状态 `rejected` |
| S1.A5 | 夹具提交非法参数 | `invalid_input`，处理函数未被调用。不代替真实 CLI 成功 |
| S2.A1 | Grok：项目 MCP、命令、读文件；另一次 `toolPolicy` 与工具并存 | 清单的启动命令不一致，或工作区 `.grok/config.toml` 声明了 MCP，则 `policy_mismatch` 且无模型进程；另两类无越权效果；冲突约束在启动前拒绝 |
| S2.A2 | Pi：扩展、命令、读文件；另一次冲突约束 | 扩展没有写出文件；另两类无越权效果；冲突约束在启动前拒绝 |
| S3.A1 | Grok：调用尚未应答时杀死宿主 | `unknown`，CLI 已退出，调用仍为 `pending` |
| S3.A2 | Pi：同 S3.A1 | 同 S3.A1 |
| S4.A1 | Grok：同一轮两个工具，默认串行 | 两次处理不重叠 |
| S4.A2 | Pi：同 S4.A1 | 同 S4.A1 |

## 验证方法

- `npm run test:execution`：夹具协议、启动前拒绝、宿主死亡、项目配置冒充、同一配置目录并发、清单读失败时不确认停止。不证明真实 CLI。
- `MILKIE_LIVE_TOOLS=1 ./node_modules/.bin/tsx tests/e2e/agent-cli-tools.live.ts`：真实 Grok 与 Pi 的 S1.A1–S1.A4、S2.A1–S2.A2、S3.A1–S3.A2、S4.A1–S4.A2。

## 已知缺口

续接时重装工具和未应答调用的后续处理属于 #267。本功能不声称模型一定会按提示词的文字顺序发起调用；串行的判定是宿主侧的处理不重叠，并且夹具能证明提交顺序被保持。
