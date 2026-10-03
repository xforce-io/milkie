# 续接宿主工具并核对未回复调用

## 用户入口

SDK `ExecutionClient.start`、`ExecutionClient.toolCall`、`ExecutionClient.pendingToolCalls`、`ExecutionClient.reconcile`。无新增 HTTP、CLI 或页面。设计为 [product.md](../../../../docs/design/267-resume-host-tool-reconcile/product.md) v1，S1–S2 的子项归本功能。

## 源码依据

`src/execution/ExecutionClient.ts`、`hostTools.ts`、`worker.ts`、`store.ts`、`types.ts`。

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| S1.A1 | 夹具：首轮 alpha+beta，新客户端去掉 alpha 后再调用 alpha | `rejected`；原生会话不变；`callId` 不同。不代替真实 CLI |
| S1.A2 | Grok：两轮、一次进程重启、收紧工具 | 原会话；撤销工具被拒绝 |
| S1.A3 | Pi：同 S1.A2 | 同 S1.A2 |
| S2.A1 | 夹具：副作用后宿主消失，核对前 start，核对后续接 | `pending` 可查询；核对前 `context_busy`；副作用只有一次 |
| S2.A1 | 夹具：宿主已返回结果，但工具连接在回复写入前断开 | 记录保持 `pending` 且没有 output；核对前 `context_busy`；`reconcile` 可接受 |
| S2.A2 | Grok：同 S2.A1 的真实执行 | 同 S2.A1 |
| S2.A3 | Pi：同 S2.A2 | 同 S2.A1 |

## 验证方法

- `npm run test:execution`：S1.A1、S2.A1。不证明真实 CLI。
- `MILKIE_LIVE_RESUME=1 ./node_modules/.bin/tsx tests/e2e/agent-cli-resume.live.ts`：真实 Grok 与 Pi 的 S1.A2–S1.A3、S2.A2–S2.A3。

## 已知缺口

业务效果去重由宿主测试自己根据已核对记录完成。milkie 不阻止已登记工具的第二次调用。
