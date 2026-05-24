---
title: agent-docs-qa example — 三国 Q&A agent with skill-loading + live trace UI
status: approved
created: 2026-05-24
related:
  - docs/superpowers/specs/2026-05-24-phase4-nondet-log-design.md
  - docs/stories/s-010-skill-versioned-load-and-ab-experiment.md
  - ARCHITECTURE.md#agent-runtime
---

# agent-docs-qa example — 三国 Q&A agent with skill-loading + live trace UI

## 1. 目标与边界

milkie 走到 Phase 1–4 完成、`Skill epoch loading` 在单测里 green、HTML report 静态 trace 渲染落地的节点。**没有任何 end-to-end example 实际用过 skill loading**；trace 实时可观察的能力也只在 HTML report 静态后置场景下存在。

本 example 兑现两件事：

1. **Dogfood**：建一个 example 作者自己日常会用的 Q&A agent（局部古典名著节选 corpus，问"赤壁双方主帅是谁"这类问题），不是 toy demo。
2. **Agent runtime 活体验收**：通过 skill_request → epoch boundary → 新指令载入 这条链路，端到端验证 milkie 的渐进式披露能力。trace 是验证手段——用户在 UI 上能**看到** `loadedSkills` 从 `['researcher']` 变成 `['researcher', 'verifier']`。

### 边界（明确不做的）

- **不做 fork / diff / suite UI 视图**——Phase 5 substrate 都没出来，UI 留口子但不实现
- **不做历史 run 的"继续聊"**（resume 已结束 conversation）—— resume 是 milkie 已有但今天 UI 不接入；要继续必须新建 chat
- **不做 skill registry / 动态 skill_list 实现**——`skill_list` 工具今天是 stub，example 不去 fix；agent base instructions 里硬编码可用 skill 名字
- **不做 RAG / embedding**——纯 grep + read 工具，trace 信息密度优先
- **不做生产级 chatbot 形态**（多用户 / 鉴权 / 跨设备同步 / 历史搜索）——单机 server，浏览器打开 `localhost:PORT` 用
- **不做第二个 on-demand skill**——一个 verifier 足够 demo 渐进式披露，多了反而被"何时该触发哪个 skill"的 LLM 决策不可靠性拖累

## 2. 设计决策（已 sign-off）

| 决策 | 选择 | 理由 |
|---|---|---|
| Agent domain | Local docs Q&A | 用户实际会用的场景；trace 表达力高 |
| 检索机制 | 纯工具调用（grep / read_file / list_dir） | 简单技术栈；trace 显示多轮推理过程；零 embedding 依赖 |
| Corpus | 三国演义 3-5 回节选，vendored 进 example 目录、frozen | public domain（中国法 1763 年的作者）；代入感高；不随主仓 docs 变化失效 |
| Skill 设计 | base "三国-researcher" + on-demand "verifier" | 一个 on-demand 足够 demo skill_request → epoch → loadedSkills 变化；用户显式触发（"verify"/"你确定"）比 LLM 自决触发可靠 |
| UI 布局 | 左 chat + 右 trace timeline + 右下 payload detail | 行业惯例；trace 实时增长用户能直接看 |
| Server 栈 | Bare Node `http` + SSE + vanilla HTML/JS | 零 framework 新依赖；example "5 分钟开箱即跑"；和 HTML report 现有风格一致 |
| Live update | SSE（server-sent events） | 浏览器原生 EventSource，server 端 trivial；单向 push 正好匹配；远胜文件轮询 |
| Persistence | active chat 单 session；past run traces 持久（JSONL 在磁盘） | example 不做产品级聊天；trace 持久 + 历史可查是 day-1 不是 follow-up |
| Conversation 颗粒 | 按 contextId 分组 N 个 runId 的事件流 | 一次对话 = 一个 contextId = N 个 invoke；UI 把同 contextId 的多 runId 合并成"一条对话线索" |
| 多 trajectory UI hook | TraceTimeline 组件参数化 runId；`?run=` URL state；`/runs` + `/trace/:runId` REST endpoints | Phase 5 fork/diff 出来时 UI 加 `?compare=A,B` 不需要重写底层 |

## 3. 架构与数据流

```
浏览器                                  Node server (single .ts file)
─────────                              ─────────────────────────────
loadIndex/conversations 列表           GET /conversations
       ─────────────────────────────►  ── scan .milkie/runs/, group by contextId
       ◄─────────────────────────────  [{contextId, agentId, startedAt, runIds}]

切换/选中某 conversation               GET /conversation/:contextId/events
       ─────────────────────────────►  ── read all runIds for contextId, merge by ts
       ◄─────────────────────────────  full events array (REST one-shot)

进入 active chat                       GET /conversation/:contextId/stream  (SSE)
       ─────────────────────────────►  ── subscribe to BroadcastingEventStore
       ◄═══════════════════════════════  push every appended event for this contextId
       (持续 stream)                     (run-completed 不 close stream;
                                         下一条 user msg 可能又来个 run)

发送 user 消息                         POST /chat  { contextId, input }
       ─────────────────────────────►  ── Milkie.invoke({agentId, contextId, input})
       ◄─────────────────────────────  { runId, status, output }
                                       agent 内部 invoke 期间，事件经
                                       BroadcastingEventStore append 同时:
                                         · 写 JsonlEventStore (持久)
                                         · 推 SSE subscribers (实时)

显式新 conversation                    POST /chat 时 contextId 留空，server 生成新 uuid
                                       返回 { contextId, runId, ... }，UI 更新 URL
                                       ?context=<新id>
```

**关键组件**：

- **`BroadcastingEventStore`**：实现 `IEventStore`，包一个 inner store（JsonlEventStore），`append()` 既走 inner.append 又广播给所有当前订阅的 SSE 连接（按 contextId 过滤——sub 只收到自己 contextId 的事件）。
- **`Milkie` 实例单例**：server 启动一次，注册 sanguo-researcher agent，整个进程共享。
- **`stateStore`**：MemoryStore（example 重启 = 清空所有进行中的 contextId 状态；JSONL traces 留在磁盘）。生产级应该用 SQLiteStore；example 不做。
- **`eventStore`**：BroadcastingEventStore(JsonlEventStore('.milkie/runs/'))。
- **`Milkie.invoke({ contextId })`**：同 contextId 多次 invoke = 多轮对话；每次 invoke 生成一个新 agentRunId，事件流入 contextId 的 broadcast channel。

## 4. 新增组件

### `examples/agent-docs-qa/` 目录布局

```
examples/agent-docs-qa/
├── README.md                       # 怎么跑 + 设计说明 + 推荐试问
├── .gitignore                      # .milkie/ runs / state，但 corpus 进版本
├── corpus/                         # 冻结的 3 国节选
│   ├── chapter-01-桃园三结义.txt
│   ├── chapter-37-三顾茅庐.txt
│   ├── chapter-49-赤壁借东风.txt
│   ├── chapter-50-华容道.txt
│   └── chapter-66-单刀赴会.txt
├── agents/
│   └── sanguo-researcher.md        # AgentConfig md：base skill + skills 表预声明 verifier
├── server.ts                       # bare Node http server (single file, ~250 lines)
├── public/
│   ├── index.html                  # 单页 HTML，inline <style> + <script>
│   └── (无其他 asset；纯 inline)
└── package.json                    # example 自己的 deps；目前 = 空（用主仓 src）
```

### `examples/agent-docs-qa/agents/sanguo-researcher.md`

```markdown
---
agentId: sanguo-researcher
version: 0.0.1
fsm:
  states:
    - name: respond
      type: llm
      instructions: |
        你回答用户关于《三国演义》的问题。Corpus 在 corpus/ 目录下，
        文件名形如 chapter-NN-标题.txt。
        - 用 list_dir 看有哪些章节
        - 用 grep 找相关章节（关键词 / 人名 / 地名 / 事件名）
        - 用 read_file 读相关段落
        - 回答时尽量给出 chapter:行号 的引用
        
        如果用户表达怀疑或要求验证（"你确定吗" / "verify" / "再确认"
        等），调用 skill_request('verifier') 进入下一 epoch 的严格
        验证模式，然后在当前 turn 告知用户"我将在下轮严格 verify"。
      tools: [list_dir, read_file, grep, skill_request]
model:
  # OpenAI-compatible adapter: supports OpenAI, Anthropic-via-proxy, DeepSeek,
  # volcengine, ollama, etc. Configure via env in server.ts (see README setup).
  provider: openai
  model: gpt-4o-mini    # default; user can override via env
  adapter: openai-compatible
skills:
  verifier: "0.1.0"
skillInstructions:
  verifier: |
    你已进入 verifier 模式。
    重新读你前一轮答案里引用过的所有原文段落，
    把每一条陈述分类：
    (a) 直接 supported by text [带 chapter:行号 citation]
    (b) inferred from text [说明推理链]
    (c) unfounded [承认错误，更正]
    严格判断，不要轻易给 (a) 标签——只要措辞和原文不严格匹配，
    应当退到 (b) 或 (c)。
---
你是《三国演义》的研究助手。
```

### `examples/agent-docs-qa/server.ts`

入口签名（示意，最终代码以 plan 步骤为准）：

```typescript
const PORT       = Number(process.env.PORT ?? 7878)
const EXAMPLE    = __dirname
const RUNS_DIR   = path.join(EXAMPLE, '.milkie', 'runs')

// 1. 启动 Milkie 实例（singleton）
const eventStore = new BroadcastingEventStore(new JsonlEventStore(RUNS_DIR))
const milkie     = new Milkie({
  stateStore: new MemoryStore(),
  gateway:    createGateway(/* selected provider */),
  eventStore,
})
milkie.loadAgentFile(path.join(EXAMPLE, 'agents', 'sanguo-researcher.md'))

// 2. 工具实现（绑定 corpus 根目录）
const CORPUS = path.join(EXAMPLE, 'corpus')
milkie.registerTool({ name: 'list_dir',  handler: ... })  // 限制在 CORPUS 下
milkie.registerTool({ name: 'read_file', handler: ... })  // 限制在 CORPUS 下
milkie.registerTool({ name: 'grep',      handler: ... })  // 限制在 CORPUS 下

// 3. HTTP server
http.createServer(async (req, res) => {
  if (req.method === 'POST' && req.url === '/chat')         return handleChat(req, res, milkie)
  if (req.method === 'GET'  && req.url === '/conversations') return handleListConversations(res)
  if (req.method === 'GET'  && url.startsWith('/conversation/'))
                                                             return handleConversationRoutes(req, res, eventStore)
  if (req.method === 'GET'  && req.url === '/')              return serveIndex(res)
  res.writeHead(404).end()
}).listen(PORT, () => console.log(`agent-docs-qa playground at http://localhost:${PORT}`))
```

`BroadcastingEventStore` 内部维护 `subscribers: Map<contextId, Set<Response>>`。`append(event)` 走 inner + 通过 event 的 runId 查 contextId（首次 lookup 后 cache），把事件 push 给对应 contextId 的所有 sub。

工具实现注意：`list_dir` / `read_file` / `grep` 全部**限定在 corpus 根目录下**——传入相对路径或者绝对路径都 normalize 到 CORPUS 内，越界直接 reject。防 LLM 误用工具读 example 目录外的东西。

### `examples/agent-docs-qa/public/index.html`

单页应用，inline `<style>` + inline `<script>`，无 build pipeline。

主要 UI 结构：

```html
<body>
  <header>
    <h1>milkie agent playground — 三国 Q&A</h1>
    <div class="controls">
      <select id="conversation-picker">  <!-- 从 /conversations 拉 -->
        <option value="">(active or new)</option>
        <!-- 历史 contextId 的 option -->
      </select>
      <button id="new-chat">+ new chat</button>
    </div>
  </header>
  <main>
    <section id="chat">
      <div id="chat-log"></div>
      <form id="chat-input">
        <input type="text" placeholder="问《三国演义》相关问题..." />
        <button>发送</button>
      </form>
    </section>
    <section id="trace">
      <div id="trace-timeline"></div>
      <div id="payload-detail"></div>
    </section>
  </main>
</body>
```

JS 行为：

- 加载时 fetch `/conversations`，填 picker
- URL 解析 `?context=<id>`；若有，加载该 conversation；否则等用户开 new chat
- 进入 conversation：
  - REST GET `/conversation/:id/events` 拿历史
  - 渲染 chat log（从 lifecycle events 重建 user/assistant turns；从 llm.responded 的最终 text content 拿 assistant 回复；user 输入从某种 metadata 里拿——见下）
  - 渲染 trace timeline
  - 如果 conversation 还活着（最近一个 runId 没 agent.run.completed），开 SSE `/conversation/:id/stream`，新事件追加到 trace 和（如果是 assistant 回复）追加到 chat
  - 历史完结的 conversation：输入框 disabled，提示 "this conversation has ended; click + new chat to start fresh"
- 用户提交输入：POST `/chat`，包含 `{ contextId, input }`；server 返回 runId 后开始 SSE 等增量

**用户输入怎么进 trace 还原**：milkie 现在 `agent.run.started.payload.input` 字段记录了 input。UI 从这里读用户原始输入恢复 chat log。assistant 回复从 `agent.run.completed.payload.lastTextOutput` 字段读。

## 5. UI 表现规范

### Trace timeline 渲染

复用 HTML report 既有的 tree.ts + html.ts 设计：

- 每个 runId 一个 section（按 startedAt 排）
- section 内列出 entries：lifecycle / llm / tool / clock.read / uuid.generated
- 默认收起 nondet（clock / uuid 在 demo 里太吵）—— filter chip "show nondet" toggle
- 点击 entry 展开 payload JSON（重用 `payloadFor` 模式）
- **关键 highlight**：当某事件触发了 `loadedSkills` 变化（agent.run.started 或 working_memory snapshot 里 loadedSkills 数组变了），整行 highlight 加 badge"skill loaded: verifier"——这是 demo 的灵魂事件

### Diagnostic 视图

- 点 trace 里任意 entry → payload 详情显示在右下
- 用户能直接看到 LLM request 的完整 messages（包含 system prompt，自然包含 base + verifier 拼接后的指令）
- 这样用户能在 UI 里直接验证"verifier 真的载入了"——前一 epoch 的 LLM request 没有 verifier 指令，后一 epoch 的 LLM request 有

### 自动滚动 + 渐入动画（可选）

- 新事件追加到 trace 时滚动到底（除非用户已经手动滚到上面）
- 新事件渐入（CSS transition）—— 让 live 增长视觉上明显

## 6. 不变式（测试用）

1. **多 invoke 同 contextId 续接**：连续两次 `milkie.invoke({contextId: X})`，第二次的 LLM messages 包含第一次的对话历史
2. **skill_request → epoch loadedSkills 变化**：record 一次 conversation，让 user 说 "verify"；事件流中找到 tool.requested (skill_request) → 下一个 agent.run.started 或 working_memory snapshot 的 loadedSkills 已含 verifier
3. **verifier 指令真的进了 system prompt**：检查载入 verifier 后第一次 llm.requested 的 payload.request.messages，system message 应包含 verifier 的指令文本
4. **BroadcastingEventStore 既写又推**：unit 测试 mock 一个 inner store + 几个 sub，append 后 inner 收到 1 次、每个 sub 收到 1 次
5. **SSE 按 contextId 隔离**：两个 conversation 并发，sub A（contextId=X）只收到 X 的事件，sub B（contextId=Y）只收到 Y 的事件
6. **工具沙箱**：list_dir / read_file / grep 试图访问 corpus 外路径 → reject

## 7. 测试策略

### Unit

- `BroadcastingEventStore` 双写 + 按 contextId 路由 sub 的 unit test
- 三个工具（list_dir / read_file / grep）的沙箱测试（输入 `../../..` 类越界路径应 reject）

### Integration

- `multi-turn-same-context.test.ts`：mock gateway，两个用户 turn，验证第二轮 LLM request messages 含第一轮 history
- `skill-loading-via-request.test.ts`：mock gateway，第一轮回普通答案，第二轮 LLM 输出包含 skill_request("verifier") 工具调用 → 验证下一轮 LLM request system 含 verifier 指令

### E2E（手动验证）

- 启动 server，浏览器访问，提一个 corpus 内问题（"赤壁双方主帅"），看 trace 出现 grep → read → llm.responded
- 跟问 "你确定吗"，看 trace 出现 tool.requested(skill_request) + 下一 epoch 的 loadedSkills 含 verifier + LLM request 的 system 有 verifier 文本
- 切换 conversation picker 到历史 conversation，看输入框 disabled、trace 完整展示历史

不写自动化 E2E（headless browser）—— example 的 e2e 价值在手动看 UI 表现，自动化覆盖 unit + integration 足够。

## 8. 落地次序

1. **Corpus vendor + .gitignore**：从公有领域三国演义来源下载 → 节选 5 回 → 放进 `corpus/`
2. **AgentConfig md**：`agents/sanguo-researcher.md`，含 base + verifier
3. **工具实现 + 沙箱**：`tools/corpus-tools.ts`（list_dir / read_file / grep + path normalize 防越界）
4. **BroadcastingEventStore**：`broadcast-event-store.ts`，unit test 先行
5. **HTTP server skeleton**：`server.ts` 起 8 个 route handler stub
6. **POST /chat 实现**：调 milkie.invoke，返回 runId
7. **GET /conversations 实现**：扫 .milkie/runs/，按 contextId 分组
8. **GET /conversation/:id/events 实现**：读全部相关 runId 事件，merge by ts
9. **GET /conversation/:id/stream 实现**：注册 SSE 订阅；连接断开时取消
10. **Frontend index.html**：HTML skeleton + conversation picker fetch + EventSource setup
11. **Frontend chat 渲染**：从 events 重建 chat log；输入框 wire 到 POST /chat
12. **Frontend trace 渲染**：复用 tree.ts / 一段简化的 timeline 渲染 + payload click handler + loadedSkills highlight
13. **Skill loading 高亮 + nondet 默认隐藏 filter**
14. **README**：跑法 + 推荐试问 + 设计说明
15. **手动 E2E 走一遍**：启动 + 真问几个问题 + 触发 skill load + 切换历史 conversation

## 9. 范围外 / 未来扩展

- 多 trajectory 比较视图（`?compare=A,B`）—— Phase 5 fork/diff land 后加，UI 已留 hook
- 继续 conversation（resume）—— milkie 已有能力，UI 加按钮接入
- skill_list 工具真实实现 + skill registry —— example 不依赖；s-010 story 真做时再补
- SQLiteStore + 跨重启 active state —— 改一行就行，但跟 example "干净 demo"目的不符
- 多 agent 切换（不只 sanguo-researcher）——header 加 agent picker
- 用户能手动加载 / 卸载 skill（不通过 LLM 决策）——后台调试 mode
- Headless browser E2E test —— Playwright 之类，复杂度收益不划算

---

## 附录：corpus 选章建议

按"角色 / 事件 / 地点交叉引用密度"挑：

| 回 | 标题 | 关键人物 | 关键事件 |
|---|---|---|---|
| 1 | 桃园三结义 | 刘备 关羽 张飞 | 起兵讨黄巾 |
| 37 | 三顾茅庐 | 刘备 诸葛亮 | 出山 / 隆中对 |
| 49 | 赤壁借东风 | 诸葛亮 周瑜 曹操 | 火攻准备 |
| 50 | 华容道 | 关羽 曹操 | 释曹 |
| 66 | 单刀赴会 | 关羽 鲁肃 | 谈判 |

5 回覆盖蜀魏吴三方主要人物、关键战役、性格刻画段落，文件总大小预估 50-80KB（中文 utf-8）。

来源：**维基文库**（公有领域中文版本，`https://zh.wikisource.org/wiki/三國演義`，繁体）—— 各回有独立页面，可直接取正文段落；如果繁简偏好不一，可走 OpenCC 转简体后入库。Vendor 时只留小说正文，去掉版本注释 / 校勘信息 / wiki markup。

入库文件命名规范：`chapter-<两位回数>-<标题关键词>.txt`，全 UTF-8、LF 换行，单文件 10-20KB 量级。

**版权安全说明**写进 example README："Corpus 节选自维基文库公有领域版本《三國演義》，作者罗贯中（约 1330-1400），全球范围内已进入公有领域。"
