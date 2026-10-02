# CLI 宿主工具技术设计

| 项 | 值 |
|---|---|
| Issue | [#266](https://github.com/xforce-io/milkie/issues/266) |
| L1 | [product.md](product.md) v1 |
| 版本 | v1 |
| 验收 | S1.A1–S1.A5、S2.A1–S2.A2、S3.A1–S3.A2、S4.A1–S4.A2 |

## 1. 设计依据与技术目标

落实 L1 v1：宿主登记工具并自己返回结果；milkie 生成调用标识、关掉未登记能力，并在宿主进程消失时终止 CLI。不实现 #267 的续接重装。

## 2. 现状与改动范围

#265 之后，CLI 使用专用目录，监督进程和 CLI 都以 `detached: true` 启动，宿主退出后执行继续。约束只有 `toolPolicy` 和 `timeoutMs`。Grok 的只读模式会排除一批内置工具；标准模式会打开权限绕过。Pi 用 `--tools` 列出内置工具。

本票在出现 `tools` 时改这条路径：监督进程不再脱离宿主，CLI 也不再脱离监督进程。没有 `tools` 的执行保持原来的脱离行为。

## 3. 总体架构与关键路径

宿主进程持有处理函数。监督进程监听一个本机 Unix socket，并把调用用已有 IPC 交给宿主。Grok 只通过专用配置目录里的一个 MCP server 连接该 socket。Pi 只加载本次写出的一个扩展，扩展再连接该 socket。

宿主进程同时握住一条管道的写端。宿主进程消失后管道读到 EOF，IPC 也会断开。监督进程据此终止 CLI，把执行写成 `unknown`，不删除尚未应答的调用记录，也不释放上下文占用。

Grok 在拉起模型前执行 `grok inspect --json`。MCP server 必须恰好是 `milkie`，且清单里的 `target` 必须等于本次写入的启动命令。工作区 `.grok/config.toml` 里只要有 `mcp_servers` 表就拒绝：Grok 1.0.41 会用它覆盖同名服务器的 command，而清单里的 source 路径仍可能指向宿主文件；该版本的清单不包含 args。外部导入单元全部关闭，托管配置处于关闭，hooks、plugins、lsp 与 marketplace 为空。skills 只允许 Grok 自带的 bundled 来源；项目或导入的 skill 失败。内置 agent 可以出现，因为命令行同时带 `--no-subagents`。不一致抛出 `policy_mismatch`，不拉起模型。

## 4. 数据与状态契约

`calls/<callId>.json` 是工具调用记录，版本为 1。字段是调用标识、可选的原生调用标识、名称、参数、`runId`、`contextId` 和状态。状态为 `pending`、`succeeded`、`invalid_input` 或 `rejected`。成功才有 `output`，失败才有 `message`。

不写入命令行、环境变量、凭据或 CLI 原始事件。参数和输出有长度上限，超出的处理结果记为 `rejected`。

Grok 的 `config.toml` 只在文件不存在，或首行已是 milkie 标记时覆写。已有其它内容则 `policy_mismatch`，不改那个文件。同一专用目录同时只允许一轮宿主工具执行持有该文件。锁的持有者仍存活，或持有者已退出但那一轮没有确认 `stopped`，后一轮都在写配置前以 `policy_mismatch` 结束。只有上一轮记录已经确认停止时，下一轮才接管残留的锁。

## 5. 接口与协作契约

`start(contextId, input, constraints, handler)` 在 `constraints.tools` 非空时要求处理函数。`forwarding` 缺省为 `serial`，只在有工具时合法。`toolPolicy` 与 `tools` 同时出现是 `unsupported_constraint`。API 传输传入工具也是 `unsupported_constraint`。工具名必须匹配 `^[a-z][a-z0-9_]{0,40}$`，且不能与已知的 Grok 或 Pi 内置工具重名。

处理函数返回 `{ ok: true, output }` 或 `{ ok: false, code: 'invalid_input' | 'rejected', message }`。返回值不合法时记为 `rejected`。

socket 上的一帧是一行 JSON：`{ id, name, nativeCallId?, input }`。回应是 `{ id, ok, output? , code?, message? }`。`id` 是 CLI 侧的关联标识。milkie 的 `callId` 另生成。Pi 扩展把 Pi 的 `toolCallId` 放进 `nativeCallId`。Grok 的 MCP 桥不填该字段。

串行时，监督进程等上一次处理函数返回后才把下一次交给宿主。并行时两次可以重叠。Pi 扩展的 `executionMode` 与该策略一致；顺序仍以监督进程的队列为准。

Grok 宿主工具命令使用 `bypassPermissions`。把内置工具从模型请求里清空时，宿主 MCP 工具也不会进入该请求，所以宿主模式不这样做。Grok 1.0.41 只通过 `search_tool` / `use_tool` 调用 MCP 工具；去掉这两个 meta-tool 后，宿主工具也不会被调用，所以宿主模式保留它们。越界访问靠只允许 `milkie` 这一台 MCP 服务器，以及 `--deny` 拦住命令、读取、编辑和搜索。`x_search`、`web_search`、`web_fetch` 用 `--disallowed-tools` 去掉。MCP 回应是一行一个 JSON；Grok 不接受 `Content-Length` 帧。环境里关闭 Cursor、Claude、Codex 的导入开关，以及托管配置开关。leader socket 仍在专用配置目录内。`HOME` 仍是该配置目录。

Pi 命令保留 `--no-extensions`，并额外传入本次扩展路径、`--no-builtin-tools` 和只含登记名称的 `--tools`。配置目录里的 `settings.json` 如果带有非空 `packages`，启动前拒绝，避免旧配置安装额外扩展。扩展连到宿主 socket 后，没有待处理调用时不再占住进程，`--print` 可以在回合结束后退出。

## 6. 失败与边界

| 条件 | 结果 |
|---|---|
| 未知约束、`toolPolicy` 与工具并存、无工具时指定转发策略、API 传入工具 | 启动前 `unsupported_constraint`，没有执行标识 |
| 缺少处理函数、工具名为空或与内置工具重名、schema 不是对象子集 | 启动前 `invalid_request` |
| Grok 清单不一致、工作区项目配置声明了 MCP、专用目录里已有非 milkie 的 `config.toml`，或该目录已有一轮存活的宿主工具执行 | 执行 `failed` / `policy_mismatch`，模型进程不出现 |
| 停止时进程清单读不到 | 执行保持 `unknown`，不把停止记为已确认 |
| 参数不匹配 | 调用 `invalid_input`，不调用处理函数 |
| 未登记的工具名或处理函数拒绝 | 调用 `rejected` |
| 宿主进程消失 | 执行 `unknown`，CLI 被终止，未应答调用保持 `pending`，占用不释放 |

错误文本只有错误码。清单输出不写入执行记录。

## 7. 验证

`npm run test:execution` 覆盖夹具协议：参数失败不调用宿主、拒绝与成功可区分、串行不重叠、并行可重叠、Grok 清单不一致时不产生 CLI 进程、宿主被杀死后状态为 `unknown`。这些不代替真实 CLI 的成功验收。

`MILKIE_LIVE_TOOLS=1 ./node_modules/.bin/tsx tests/e2e/agent-cli-tools.live.ts` 对真实 Grok 与 Pi 覆盖 S1.A1–S1.A4、S2.A1–S2.A2、S3.A1–S3.A2、S4.A1–S4.A2。
