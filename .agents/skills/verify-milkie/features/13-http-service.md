# HTTP 服务与流式执行

## 用户入口

- CLI：`serve`；--agent/--port/--host/--state-store/--data-dir
- HTTP：`GET /health`、`POST /chat`；其它路由按 README 入口表分流

## 源码依据

- [src/cli/main.ts](../../../../src/cli/main.ts)
- [src/cli/serve.ts](../../../../src/cli/serve.ts)
- [src/trace/BroadcastingEventStore.ts](../../../../src/trace/BroadcastingEventStore.ts)
- [tests/e2e/fixtures/serve-stub-entry.ts](../../../../tests/e2e/fixtures/serve-stub-entry.ts)

## 驾驶路径与判定

| 路径 | 操作 | 可判定结果 |
|---|---|---|
| 启动/请求 | 启动独立进程，等 MILKIE_SERVE_READY，GET /health，再 POST /chat | health 为 {ok:true}；活动帧、message_delta、唯一结束帧可观察，连接正常结束。 |
| 错误/断连 | 运行时错误、缺少 contextId、未知路由；客户端中途断开，再发 health | 错误帧和终态可区分；缺字段 400、未知路由 404；断连不导致服务崩溃；断连不自动等同取消。 |
| 结果一致性 | 模型持续工具调用至 max_iterations；比较 SDK、持久化结束记录与 SSE | 三者保留相同 stopReason/stopCode/partial/artifacts/checkpointId/error；SSE 保留 output/runId/contextId。 |
| 退出 | 保持 stdin 打开进行请求，再 SIGTERM 或关闭 stdin | 进程退出，无残留监听；只清理本次启动的 PID。 |

## 验证方法

下列命令从仓库根运行；是自动化覆盖入口，不代表上表所有路径已通过。具体执行记录见[首版核验](../references/initial-audit.md)。

```sh
./node_modules/.bin/jest --runInBand --runTestsByPath src/__tests__/serve.test.ts src/__tests__/serveCli.test.ts src/__tests__/BroadcastingEventStore.test.ts src/__tests__/stopResult.exits.test.ts tests/e2e/recovery-cli.e2e.test.ts
```

- [src/__tests__/serve.test.ts](../../../../src/__tests__/serve.test.ts)
- [src/__tests__/serveCli.test.ts](../../../../src/__tests__/serveCli.test.ts)
- [src/__tests__/BroadcastingEventStore.test.ts](../../../../src/__tests__/BroadcastingEventStore.test.ts)

## 已知缺口

既有确定性服务 fixture 绕过 CLI 加载；新增 recovery-cli E2E 使用真实 CLI/serve 及本地 OpenAI 协议端点，覆盖文件加载和进程路径。真实远端供应商仍未验证。结果字段修复见[本次验收入口](../references/issues-259-261.md)。

- [src/__tests__/stopResult.exits.test.ts](../../../../src/__tests__/stopResult.exits.test.ts)
- [tests/e2e/recovery-cli.e2e.test.ts](../../../../tests/e2e/recovery-cli.e2e.test.ts)
