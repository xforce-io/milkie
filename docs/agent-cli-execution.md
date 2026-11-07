# 外部执行 SDK

> 适配版本：Grok 1.0.41、Pi 0.85.1、Codex 0.166.0（#277 候选，真实验收待完成）；其它版本须重新验证。真实验收证据按候选 SHA 记录在 #263 的 PR 中。

先构建 `npm run build`。SDK 从发布物导入：

```ts
import { ExecutionClient } from '@freemanxu/milkie'

const execution = new ExecutionClient({
  dataDir: '/path/to/private/milkie-execution',
  connection: {
    contractVersion: 1,
    fields: { transport: 'agent-cli', runtime: 'pi' },
  },
})
const capabilities = execution.capabilities()
// supported 表示有适配器；availability: unchecked 表示安装/登录尚未实际验证。
const context = execution.createContext('/path/to/workspace', {
  configDir: '/path/to/dedicated-pi-config',
  sessionDir: '/path/to/dedicated-pi-sessions',
})
const runId = execution.start(context.contextId, '解释当前目录的 README', {
  toolPolicy: 'read-only',
  timeoutMs: 120000,
})
const result = await execution.wait(runId)
```

保存 contextId 和 runId。宿主重新进入时用同一 dataDir、同一连接重建客户端；`start(contextId, 新输入)` 续接，`createContext` 明确新建。`getContext` 可查原生会话关联，`query(runId)` 可查本轮结果。`cancel(runId)` 请求监督进程停止所属执行，返回停止确认或 unknown。

Grok、Pi 与 Codex 提供执行适配；Claude Code 仅保留配置兼容，执行返回 unsupported_runtime。Codex 必须显式指定模型，原生 CLI 版本须精确为0.166.0，缺失或不匹配分别返回 runtime_missing、unsupported_version；capabilities.availability=unchecked 不表示已安装或已登录。创建 CLI 执行上下文时必须给出专用配置目录和专用会话目录，目录或登录材料缺失时在启动前返回 `config_missing` 或 `session_missing`，不会改读宿主 HOME 下的 CLI 配置，也不会连接 Grok 的默认共享 leader。登录材料由宿主放入配置目录的 `auth.json`；milkie 不复制凭据，也不把宿主进程中的供应商密钥传给 CLI。Pi 的扩展、技能、模板、上下文文件发现均关闭，以免外部资源扩大工具权限。Grok 使用精确 UUID 恢复、权限模式与工具排除参数；默认只读会拒绝工具写入；Grok 的权限拒绝可能返回 failed/native_cancelled，不能把 CLI 的零退出码当作任务成功。

实现依据见 [设计](design/265-agent-cli-dedicated-storage/product.md)。下面的执行生命周期仍以 [263 设计](design/263-agent-cli-execution.md) 为准。

API 使用同样的 SDK，连接填 `transport: api`、protocol、model 和 apiKey。单次调用复用既有 gateway；不提供 CLI 原生会话或工具循环语义。凭据通过 IPC 交付，不写入上下文或执行记录。显式最终 output 是业务结果，调用方自行决定保存和展示；日志不包含原始事件或诊断文本。

`standard` 仅由 Pi/Grok 支持；Codex 启动前拒绝此约束，原生工具始终关闭，读写能力只能来自宿主登记。对 Pi/Grok，`standard` 是调用方显式授权原生工具读写与命令执行：Pi 开放指定内置工具，Grok 本轮关闭原生逐工具确认。它不提供路径隔离，不应交给未授权调用方；默认 `read-only` 不启用此模式。

宿主工具与 `toolPolicy` 不能同时使用。宿主传入工具名称、说明和 JSON Schema 对象子集，并提供处理函数。milkie 生成调用标识，把调用交给该函数，再把结果交回 CLI。参数不匹配是 `invalid_input`，宿主拒绝是 `rejected`，二者都不是成功。未指定转发策略时按串行交给宿主；需要并行时显式传入 `forwarding: 'parallel'`。这种执行不脱离宿主：宿主进程消失后，CLI 被终止，执行状态为 `unknown`，尚未应答的调用记录保留。下一轮可以去掉部分工具；不在本轮授权表里的调用记为 `rejected`。CLI 目录仍保留该上下文曾经登记的名称，这样调用才能到达入口。还有 `pending` 调用时，`start` 返回 `context_busy`。`pendingToolCalls(contextId)` 列出这些调用，包括尚未交给处理函数的队列项。宿主用 `reconcile(callId, output)` 记下核对结果，并自行把这段结果写进下一次 `start` 的输入；全部核对完才释放占用并续接原会话。Grok 在此模式下只加载 milkie 登记的一个 MCP server，并在模型启动前核对实际加载结果；不一致时执行失败码是 `policy_mismatch`。Pi 只加载本次生成的扩展，`--tools` 只有宿主登记的名称。能力里的 `nativeCallId` 表示该 CLI 会不会提供自己的工具调用标识：Pi 与 Codex 会，Grok 不会。详见 [宿主工具设计](design/266-agent-cli-host-tools/product.md)。

默认只读，执行时限 120 秒、最大一小时；未知约束在启动前拒绝。工作目录不等于文件系统隔离。不能把停止视为撤销文件或远端副作用。监督进程核对继承本轮标识和已观察到的子进程，包含脱离原进程组的任务；不能确认时返回 unknown。此机制用于可信任务，不是对抗性内核隔离，也不覆盖外部已有服务；要求未声明的隔离约束会在启动前拒绝。

Pi/Grok 可以在单次 `start` 上传入 `maxModelIterations`（整数 1 到 10000）。这是本轮允许的模型迭代次数，不是工具调用次数，也不是超时；不传则保持原有行为。Grok 使用 `--max-turns`，Pi 在发往提供方之前中止超额请求。用尽后执行状态为 `failed`，失败码是 `iteration_budget_exhausted`，同一上下文的下一次 `start` 仍续接原会话。API 传输及 Codex 拒绝该约束，且不会发出模型请求。Pi 在设置预算时关闭会绕过计数的自动及手动压缩，并把提供方内部重试设为零；Pi 上层重试仍经过计数。过长会话可能返回上下文容量错误。宿主工具的原始字符串和成功结果以 UTF-8 字节计，上限 256 KiB；编码后的单次工具请求上限 2 MiB。超限是明确拒绝，不截断，合法结果也不会变成 uncertain。核对结果同样以 256 KiB 字节为上限。详见 [275 设计](design/275-cli-iteration-and-tool-bounds.md)。

`starting/running` 表示活动执行；`succeeded/failed/cancelled/timed_out` 是已确认本地资源停止的终态。心跳超过 5 秒未更新会查询为 unknown；等待超时返回最新已知状态，不取消执行或伪造终态；取消超过核对时限则返回 unknown。unknown 不自动回收占用或重放。尚未核对的宿主工具调用继续占用上下文；核对完成后，下一次 start 续接原会话。未核对时不要新建上下文来掩盖失败。稍后仍可查询或取消原 runId。API 取消只确认本地请求结束，不保证远端计算停止。

实现依据见 [设计](design/263-agent-cli-execution.md)。


## 本地 Sidecar 与 Codex

Rust 等宿主可启动 `milkie execution-sidecar --data-dir /absolute/data --connection-file /absolute/connection.json`。连接文件沿用 ConnectionInput，宿主负责准备；不通过 argv 传递凭据。stdout 仅承载 JSON-RPC 2.0 JSONL，先 initialize 核对 protocolVersion=1，再创建上下文并启动/查询/取消执行；工具通过反向 tool.call 请求返回宿主。完整方法、边界和错误见 [277 契约](design/277-codex-sidecar-execution.md)。

Sidecar 使用 `lifetime: 'host'`，即使不登记工具，宿主失联也会停止原生进程。SDK 可显式选此模式；默认 independent 保持原有无工具执行的独立生命周期，带工具的执行始终依赖宿主。stdout 故障、EOF、SIGTERM 和强杀均不把未知效果改写成成功。

Codex 的 CODEX_HOME 使用宿主专用配置目录，sessions 绑定专用会话目录；原生 thread id、JSONL 路径及 session_meta id/cwd 必须一致。0.166 的续接恢复最初工具目录；同一上下文只能保留或撤销原有定义，增加工具或改变定义返回 unsupported_constraint。撤销不等于从历史中删除名称，调用仍由本轮授权校验拒绝。要改变工具目录应明确创建新上下文，不能悄悄切换原生会话。

Codex 适配经 app-server initialize/initialized、thread/start 或 thread/resume、turn/start 驱动；原生 MCP、shell、web、插件与 Agent 能力启动前关闭，未登记请求拒绝。工具成功以结果写回原生管道为送达依据，失败写入保留 pending；这不证明业务副作用可撤销，也不提供 exactly-once。
