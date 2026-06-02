# milkie serve(最小可用版)设计

- **Issue**: #86 — [alfred][P1] 最小可用 milkie serve(HTTP + SSE,含 token 流透传)
- **状态**: 已评审通过(2026-06-02,见 #86 评论)
- **分支**: `feat/86-milkie-serve`
- **关联**: alfred provider 抽象层 xforce-io/alfred#32;前置 #80/#81(已 closed);雏形 `examples/interrupt-resume-sidecar/`(#85)

本文档是该功能的 single source of truth。issue body 仅保留摘要 + 指向本文档的链接。

---

## 1. 目标与范围

为 alfred 提供一个**稳定的进程边界接口**驱动 milkie:`milkie serve --agent <file> --port <p>` 起一个被父进程(alfred)拉起、生命周期绑定的 HTTP + SSE 服务。**只做最小可用版**,产品级能力(并发/背压/鉴权/配置)后置。本质是把现有 `interrupt-resume-sidecar` 提升为正式 `milkie serve` 命令,并补上唯一缺口:token 级 `message_delta` 经 SSE 透出。

## 2. 决策摘要

| # | 决策 | 理由 |
|---|---|---|
| D1 | `serve` 作为现有 `milkie` CLI 的子命令,**不新增 bin** | milkie 已有 commander CLI(`agent`/`trace`),serve 与之平级 |
| D2 | `POST /chat` 响应体**直接是 SSE 流**;interrupt/resume 带外 | 单请求流式(OpenAI/Anthropic 范式),消除时序窗口,最贴 dolphin 进程内 async iterator 语义 |
| D3 | 工具调用走**持久化粗粒度** `tool.requested/responded`;唯一新建「非持久化→SSE」通路的是 `message_delta` | milkie 与 dolphin 同构:token 级拼装是内部细节,对外暴露的是一次性工具调用 |
| D4 | **阻塞型前台进程**,SIGTERM/关 stdin 优雅退出,**不做** start/stop 子命令 | 匹配 alfred「spawn 子进程 + 持句柄 + 发信号」托管模型;daemon 化会制造孤儿进程,违背「绑定父进程」 |
| D5 | 进程内 `MemoryStore`(interrupt/resume 同进程内够用) | 最小版不追求 serve 重启后跨进程 resume |

## 3. 进程模型(D4)

`milkie serve` 前台阻塞常驻 = 进程即服务,生命周期外包给父进程 alfred:

```
1. spawn:  child = spawn('milkie', ['serve','--agent','x.md','--port','8723'])
2. 就绪:   child.stdout 出现 "MILKIE_SERVE_READY 8723"
3. 驱动:   POST /chat(SSE)/ POST /interrupt / POST /resume
4. 停止:   child.kill('SIGTERM')  → server.close() + 清理 → 进程退出
```

不做 daemon/start-stop:那会让真正的 server detach 成孤儿,alfred 失去直接控制且违背「绑定父进程 / 无僵尸」。需要后台化时交给外部进程管理器(systemd/pm2),不进 milkie。

## 4. CLI 集成(D1)与「常驻 vs 短命令」模型兼容

`src/cli/main.ts` 现有命令是「执行即返回 `{stdout,stderr,exitCode}`」模型。serve 作为常驻命令需处理三点:

1. **就绪信号直写 `process.stdout`**:现有 CLI 把输出攒进数组、`main()` 返回时才 flush;serve 不立即返回会卡住就绪信号 → 必须绕过缓冲直写。
2. **resolve-on-shutdown**:serve action 返回的 Promise 在优雅关闭完成时才 resolve(非永挂、非硬 exit)→ `main()` 的「执行后返回 exitCode」契约依然成立,执行时长 = server 生命周期。
3. **抽出 `createServeServer(opts)`(side-effect free,只建不 listen)**:供单测;常驻行为用子进程 e2e 验证。

启动期错误(端口占用 / `--agent` 不存在)在 listen 前 throw → 走 commander 现有错误路径返回非 0;运行期错误由 server/handler 兜,不 crash 进程。

## 5. HTTP 接口契约

| 方法 路由 | 说明 |
|---|---|
| `GET /health` | `{ ok: true }` |
| `POST /chat { contextId, goal?, input }` | 响应 `Content-Type: text/event-stream`,推送本 turn 事件流,以终态事件收尾后关闭流。**`contextId` 必传**(client 生成,供 interrupt 定位 + 事件订阅) |
| `POST /interrupt { contextId }` | 带外控制,`{ signaled: true }`;milkie 在下一 yield 点观察信号,使对应 `/chat` 流以 `interrupted` 终态收尾 |
| `POST /resume { contextId, input? }` | 同 `/chat`,返回**新的** SSE 流继续 |

就绪:listen 后 stdout 打印 `MILKIE_SERVE_READY {port}`。退出:SIGTERM / 关 stdin → 优雅关闭。

## 6. SSE 事件契约(D2 + D3)

### 6.1 为什么 POST /chat 直接返回流(方案①)

dolphin 进程内是单调用 async iterator(`run_turn()` yield events),触发=消费同一个流,无时序窗口。HTTP 把它拆成两请求(异步 POST + 独立 GET /stream)才产生「事件早于订阅者」竞态。方案①(POST 触发、响应体即 SSE)= 单请求、无竞态、零服务端缓冲,且正是 alfred 团队最熟的 OpenAI/Anthropic streaming 范式。

> 备选已排除:② POST 创建 + GET 订阅 + `Last-Event-ID` 回放(需事件历史缓冲,超最小版);③ 先连后发(引入竞态 + 调用顺序约定)。

### 6.2 SSE 信封与事件白名单

信封采用标准 SSE,`event:` 用 **milkie 原生事件名**,`data:` 为该事件 JSON payload:

```
event: message_delta
data: {"text":"曹"}

event: tool.requested
data: {...}

event: agent.run.completed
data: {"status":"completed","lastTextOutput":"...","error":null}

```

> **字段命名解耦**:milkie 透出原生事件名/payload;事件映射(pid 合成、stage 分类、字段翻译到 dolphin `TurnEventType`)是 alfred 侧 xforce-io/alfred#32 适配层的职责。milkie 不绑死 alfred 协议。

透出白名单(对齐 alfred 消费的 dolphin 四类事件 + 起止/终态):

| dolphin(alfred 消费) | milkie 透出 | 通路 |
|---|---|---|
| LLM delta | `message_delta` | **新桥接**(onModelEvent) |
| tool_call | `tool.requested` | 持久化广播 |
| tool_output | `tool.responded` | 持久化广播 |
| skill | `skill.loaded` / `skill.unloaded` | 持久化广播 |
| (StageInstance 进度) | `fsm.transition` | 持久化广播 |
| 起始 | `agent.run.started` | 持久化广播 |
| 终态 | `agent.run.completed`(由 serve 基于 AgentResult **合成**,不走广播 — 见 §7.2) | serve 合成 |
| 子 agent | `agent.spawned` / `agent.returned` | 持久化广播 |

排除(不透出):`clock.read` / `uuid.generated` / `wm.mutated` / `region.*` / `context.boundary.applied` / `agent.checkpoint` / `object.created` / `relation.created`。

### 6.3 工具调用为何走持久化粗粒度(D3)

`StreamAggregator` 把 token 级 `tool_call_start/delta/done` 拼装成完整参数,聚合完成后 `AgentRuntime` 才一次性执行工具、发持久化 `tool.requested/responded`。dolphin 同构:token 级拼装是内部细节,alfred 消费的 `tool_call/tool_output` 是一次性执行事件。故 `/stream` 用持久化粗粒度即可,**无需**桥接 `tool_call_*`。

### 6.4 message_delta 桥接实现

- **持久化事件**:`BroadcastingEventStore`(实现已从 examples 提升到 `src/trace/`,examples re-export)按 `contextId` 广播(靠 `agent.run.started` 建 runId→contextId 映射,`subscribe(contextId, cb)`);`POST /chat` handler 订阅该 contextId。
- **`message_delta`**:`POST /chat` 调 `invoke` 时传入 `onModelEvent`,在 handler 闭包内(已知 contextId)把 delta 写入同一条 SSE 响应。`resume` 路径同样:`Milkie.resume` 新增可选 `onModelEvent`(与 `invoke` 对称),resume 流也逐 token 透出。
- 两路合并到同一 `text/event-stream` 响应;SSE 关闭时反订阅,防泄漏。

## 7. SSE 流的收尾 / 打断 / 错误契约(alfred 4 条硬要求)

alfred 适配层正确**收尾 / 打断 / 报错**的前提,这 4 条不能省。验证结论:milkie 现有机制**全部原生满足**。

1. **逐 token `message_delta`** — 本 issue 核心(§6.4)。

2. **明确的终态事件** — 流必须以一个带 `status` + `output` 的终态事件收尾,alfred 靠它判定本轮结束并取最终结果(也是「省 /status」成立的前提)。
   - 落地:终态由 serve 基于 `invoke`/`resume` 返回的 `AgentResult`(自带 `status` + `output`)**合成**一个 `agent.run.completed` 帧后关闭流。广播白名单**排除** `agent.run.completed`,使终态唯一且确定(不依赖广播时序)。

3. **interrupt 语义** — `POST /interrupt { contextId }` 中断正在某 `/chat` SSE 流里运行的轮次,该流以 `interrupted` 终态事件收尾,**而非裸断连**。
   - 落地:`contextId` 在 `/chat` 必传(client 生成),interrupt 据此定位运行中的 run;`milkie.interrupt()` 使 run 在下一 yield 点以 `status:'interrupted'` 结束(`AgentRuntime` 抛 `Agent interrupted`),终态事件 `agent.run.completed{status:'interrupted'}` 写入流后再关闭。

4. **error 不静默** — LLM/工具出错时,以 `error` 事件 + 终态事件出现在流里,**不静默断开连接**(否则 alfred 无法区分「出错」与「网络断」)。
   - 落地:`POST /chat` 内 `invoke` reject 时,SSE 层 catch → 写一个 `error` 事件(`{message}`)+ `agent.run.completed{status:'error',output:'',error}` 终态 → 关闭流。

## 8. 运行时与存储(D5)

- **agent**:`Milkie.loadAgentFile(--agent)`(.md frontmatter),不走 `.milkie/agents.json` manifest。
- **stateStore**:进程内 `MemoryStore`;**eventStore**:`BroadcastingEventStore(MemoryEventStore)`。interrupt/resume 在同一 serve 进程内,内存即可(同 #85 sidecar)。serve 重启后旧会话不可 resume —— 最小版接受(serve 绑定 alfred,重启即重来)。

## 9. 非目标(后置到完整版)

并发多会话上限 / SSE 背压 / 鉴权 / 配置体系 / 健康指标 / 错误语义完备 / `Last-Event-ID` 断点续传 / serve 重启后跨进程 resume 的持久化。出现第二个消费方时,再升级为产品级常驻 server。

## 10. 验收与测试

- **单元**:`createServeServer` —— 各端点、SSE 信封格式、事件白名单过滤、订阅/反订阅、终态/error 收尾。
- **子进程 e2e**(参 #85 sidecar 测法):起 serve → 就绪信号 → `POST /chat` 收 ≥2 个 `message_delta` + 终态 `agent.run.completed{status:'completed'}` → interrupt 使流以 `interrupted` 终态收尾 → resume 返回新流 → SIGTERM 优雅退出无僵尸。
- **alfred 侧**(issue 验收):Python client + 一条纯文本问答冒烟,收到 ≥2 个 `message_delta` chunk + 终态 `AgentResult`。

## 11. 落点

- 新增 `src/cli/serve.ts`:`createServeServer(opts)`(side-effect free)+ serve action。
- `src/cli/main.ts`:注册 `program.command('serve')`。
- `package.json` 的 `bin` **不变**(已是 `{"milkie":"dist/cli/index.js"}`)。
- 测试:`src/__tests__/` 单元 + `tests/e2e/` 子进程 e2e。
