# #79 服务日志（service/operational logging）

> Issue: https://github.com/xforce-io/milkie/issues/79
> 状态：设计定稿（讨论记录见 issue 评论），随 feat/79-service-logging 实现。

## 1. 定位：三层可观测，service log 是第三层

| 层 | 实现 | 视角 | 性质 |
|---|---|---|---|
| event log | `src/trace/JsonlEventStore.ts` | 业务事实（llm/tool 配对事件） | 真相源，可 replay |
| trajectory | `src/trajectory/*`（`ITrajectoryRecorder`，Span 模型） | 业务执行的因果/耗时结构 | 诊断 |
| **service log（本期）** | `src/logging/*` | 运维：服务在怎么跑 | 遥测，可丢弃 |

对齐业界三支柱：event log ≈ events、trajectory ≈ tracing、service log ≈ logging。
**三层并存，互不吸收**，用 `runId`/`contextId` 关联互跳。

**边界纪律**：service log 只在服务边界打 wide event，不进入执行内部。
判别式：*运维值班的人要看的*进 service log；排查业务执行细节才看的留 trajectory；
不把运维日志塞进 event log，也不用 event log 当运维日志读。

## 2. 选型与形态

- **pino** 作 logger 本体（运行时依赖）。不自研：这块是独立能力，会持续长大。
- NDJSON 输出到 **stderr**。偏离 issue 原文的"stdout"是有意的：stdout 已被程序输出占用
  ——CLI 的结果输出（`cli/index.ts`）、serve 的 `MILKIE_SERVE_READY <port>` 端口标记
  （下游靠解析它发现端口）。日志混入会破坏下游解析。12-factor 语义不变：仍是标准流、
  不落文件、由外部采集（本地 `2> milkie.log`）。
- `pino-pretty` 作 devDependency：TTY（`process.stderr.isTTY`）自动启用，
  `LOG_FORMAT=json|pretty` 强制覆盖；pretty 不可用（生产未装）时静默回退 json。
- `LOG_LEVEL=debug|info|warn|error|silent` 控级别，默认 `info`。

## 3. 接口（`src/logging/`）

不在 pino 外造抽象层；只导出配置好的工厂与类型别名：

```ts
export type ServiceLogger = pino.Logger          // re-export，调用方不直接 import pino
export function createServiceLogger(opts?: {
  level?: string                                  // 默认 LOG_LEVEL ?? 'info'
  format?: 'json' | 'pretty'                      // 默认 LOG_FORMAT ?? (stderr TTY ? pretty : json)
  destination?: pino.DestinationStream            // 默认 stderr；测试注入内存 sink
}): ServiceLogger
export function getLogger(): ServiceLogger        // 惰性默认单例（读环境变量）
```

维度通过 pino child 注入，埋点处不手动带字段：

```ts
logger.child({ mod: 'server' })          // 模块维度：runtime/gateway/server/cli/store/tools
logger.child({ runId, contextId })       // 请求维度：correlation id
```

## 4. 字段 schema

固定：`ts`（ISO8601 UTC）、`level`（字符串标签）、`mod`、`msg`、`runId`/`contextId`（有则带）、
`err`（pino 标准 serializer，含 `message`/`stack`）。事件附加字段平铺：

- HTTP：`method` `path` `status` `ms`
- LLM：`model` `durationMs` `inputTokens` `outputTokens`
- invoke：`agentId` `runId` `contextId` `durationMs` `status`

## 5. 埋点清单（MVP）

| 位置 | 事件 | 级别 |
|---|---|---|
| `serveMain` / `runServeServer` | 启动（port/agentId/stateStore 种类）、关闭 | info |
| `createServeServer` 请求分发层 | 每请求一条：method/path/status/ms（SSE 含完整流时长） | info |
| `Milkie.invoke()` | 每次 invoke 一条汇总：agentId/runId/contextId/durationMs/status | info / error |
| `LoggingGateway`（`GatewayFactory.createGateway` 统一包装两 adapter） | 每次 LLM 调用一条：model/durationMs/token 用量；失败带 `err` | info / error |
| 收编零散 console.*：`tools/system.ts` manifest 三处 warn、`Milkie.resolveModel` tier 回退 | `logger.warn` | warn |

埋点偏差说明：issue 原文的 `turns` 不记——`AgentResult` 未暴露 turn 数，待运行时暴露后补；
"gateway 重试"埋点无对应物（重试只在工具层，归 trajectory），本期不做。

不动的：CLI 结果输出是程序输出非日志；`trajectory/ConsoleRecorder` 属 trajectory 层。

## 6. 脱敏

service log **不携带 prompt/completion 正文**，只记元数据。正文在 event log，按 runId 跳转。
`err.message` 保留原样；不主动序列化请求体。

## 7. 测试策略

注入内存 destination 断言 NDJSON。覆盖：级别过滤、format 选择（LOG_FORMAT/TTY/回退）、
child 字段合并、err 序列化、LLM 成败两路、HTTP 请求行、并发 invoke 时 runId 不串线。
