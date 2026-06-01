# #83 会话 context 变量读/写 API（跨进程可达）

> 状态：**设计稿，待评审**。本文是 single source of truth（issue #83 留摘要 + 链接）。
> 关联：[[#82 per-turn variables]]（已合并，本设计复用其变量模型）、#86（server，仅作 HTTP 门面，非依赖）、#84（会话导出/导入）。

---

## 1. 背景与问题

alfred 用 milkie 替换 dolphin provider。dolphin 里，后台模块（heartbeat / cron / workflow）**直接读写** `agent.executor.context` 的变量（`session_id` / `query` / `workspace_instructions` / `session_created_at`）。

替换后，对话状态住进了 milkie（Node 进程）的内部链路：

- 会话状态的真正载体是 **event log + checkpoint 投影**（`checkpointFromEvents`）——这是一个**只读投影**，外部进程改不动；
- `WorkingMemory` 是 **agent 自己**写的内部 memory，也住在 checkpoint 里，外部同样碰不到；
- `IStateStore` 目前只有 4 个约定 key（`checkpoint-run:latest` / `interrupt` / `children`），**没有任意 KV 层**。

→ alfred 的后台模块失去了读写会话变量的能力。这就是 #83 要解决的：**让会话级变量脱离 Node 内部链路，成为一个外部进程也能平等读写、且下次 invoke 时 agent 能看到的东西。**

### 1.1 为什么 #82 不够、必须 #83（带外 / 跨时机）

#82（per-turn variables）的前提是 **invoke 调用方**在调用那一刻就知道并传入变量（如 `current_time`，sidecar invoke 时当场取）。

但 alfred 的后台角色（heartbeat / cron / workflow / web "停止·改设置"）**不掌握 invoke 调用点**——它们在两次 invoke 之间、在别的进程里，想让某变量"等下次对话生效"。它们没有调用点可塞 #82 的 variables，只能把变量**带外写进共享层**，等下次 invoke 自动读到。

| 维度 | #82 | #83 |
|---|---|---|
| 谁写 | invoke 调用方 | 带外角色（后台 / 别的进程） |
| 何时写 | invoke 那一刻 | 任意时刻，等下次 invoke 读 |
| 写者位置 | 与 invoke 同进程 | 通常另一进程（alfred = Python 后台） |

→ #83 的本质是"**带外修改会话状态**"；"跨进程"只是 alfred 架构（Node sidecar + Python 后台）下的具体形态。反之，单进程嵌入、调用方每轮自组装上下文的场景，#82 就够，#83 多余。

---

## 2. 目标 / 非目标

**目标**
- 提供 `getContextVar / setContextVar / deleteContextVar / listContextVars`（contextId 维度）。
- **跨进程可达**：Python 后台模块也能读写。
- 写入的变量，**下一次 invoke 时对 agent 可见**（渲染进 prompt）。
- 与 #82 的 per-turn 变量共用一套数据模型，但持久化、低频变化。

**非目标**
- 不做官方 HTTP server（那是 #86；本设计只保证"可被 #86 暴露"）。
- 不替代 `WorkingMemory`（agent 内部 memory，正交概念，§7）。
- 不做变量的版本历史 / 审计（只存当前值）。
- 不做自动 GC（生命周期见 §5.7）。

---

## 3. 前置知识：milkie 的 prompt 是怎么拼出来的（region 模型）

> 这一节是理解后面所有设计的基础。如果你已经清楚 region substrate，可跳到 §4。

milkie 发给大模型的不是"一坨字符串"，而是由若干 **region**（上下文区块）**按固定顺序拼装**而成。每个 region 有几个关键属性：

- `target`：`system`（拼进 system prompt）/ `message`（拼进 messages 数组）/ `tool`（拼进 tools）。
- `section`：在所属 target 内的**位置槽位**，顺序由 `sectionSchema.ts` 决定。
- `stability`：`immutable | session-stable | turn-stable | volatile`——这块内容**多久变一次**。
- `cacheBreakpoint`：是否在此设置 prefix-cache 断点。

**message 段的顺序**（#82 合并后 + #83 新增 session-context）：

```
history  →  [session-context]  →  turn-context  →  current-turn  →  scratchpad
（旧对话）   （#83会话变量·新增）    （#82本轮变量）   （本轮用户输入）   （工具往返草稿）
   ▲
   └─ cacheBreakpoint 设在 history 末：system+history 进 cache，受 session-context 变化保护
```

**一次 invoke 实际拼出的 prompt 长这样**（示意）：

```
┌─ system ────────────────────────────────────────────┐
│ header        : You are alfred's assistant. ...      │  ← immutable，永远命中 cache
│ persistent-skills / wm / ...                         │
└──────────────────────────────────────────────────────┘
┌─ messages ──────────────────────────────────────────┐
│ [history]      user: 上一轮问题 / assistant: 上一轮回答 │  ← session-stable
│ [turn-context] user: "--- Turn Context ---           │  ← #82, volatile（每轮重渲染）
│                       current_time: 2026-06-01T..."   │
│ [current-turn] user: "Goal: ...\n\n本轮输入"          │  ← volatile
└──────────────────────────────────────────────────────┘
```

**为什么 #82 把每轮变量放在 message 段而不是 system 段？**
因为 prefix cache 命中的是"从头到第一个变化点之前"的最长**稳定前缀**。`current_time` 每轮都变；如果塞进 system prompt 靠前位置，会让 system 前缀每轮失效，整段 cache 击穿。放在 message 段、history 之后，则**完全不碰 system 前缀**。

**#83 要新增的，就是在这套顺序里插入一个承载"持久会话变量"的 region**——叫 `session-context`。它和 turn-context 同样用 `key: value` 渲染，但**变化频率低**（session-stable）。

它放在 `history` **之后**（不是 system 段）——因为这些变量会变（如 `workspace_instructions`），而 system 段在 history 之前，把会变的内容放 history 前会击穿越来越大的 history cache。放 history 之后，变化只波及它身后的小尾巴。详见 §9 O1（已定）。

---

## 4. 核心决策：方案 Y —— 共享 store + key 契约（已拍板）

vars 存进**进程间共享的 store 后端**（Redis 或 SQLite 文件），按公开的 key 契约存放。Python 后台**直连同一个 store** 读写，不必经过 Node。Milkie SDK 的几个方法只是 Node 侧的便捷封装。

```
        ┌────────────────────┐         ┌──────────────────────┐
        │  Node (milkie)     │         │  Python (alfred后台)  │
        │  Milkie.setContextVar────┐    │  heartbeat / cron     │
        │  Milkie.getContextVar    │    │     │                 │
        └──────────┬─────────┘     │    └─────┼─────────────────┘
                   │               │          │
                   ▼               ▼          ▼
            ┌──────────────────────────────────────┐
            │   共享 store（Redis / SQLite 文件）     │
            │   key: context:{id}:var:{name}         │  ← 公开契约（§5.1）
            └──────────────────────────────────────┘
```

**这条路的两个直接后果**：
1. **`MemoryStore` 不满足 #83**——它是进程内的，跨不了进程。#83 的"跨进程可达"隐含前提：**部署时 store 必须是 Redis / SQLite**。这点要在文档和验收里写明。
2. **#83 不依赖 #86**——SDK 层 + key 契约本身就能让 Python 直连。#86 的 HTTP 端点只是给"不想直连 store 的客户端"用的门面，可后补。

### 4.1 store 选型与部署前提

| 后端 | 进程边界 | 定位 |
|---|---|---|
| `MemoryStore` | 单进程独占 | 测试 / 单进程 / 开发 |
| `SQLiteStore` | 同机多进程共享文件 | 同机持久 / 单机多进程 |
| `RedisStore` | 跨机跨进程 | **跨进程生产（alfred 推荐）** |

**硬前提**：启用 #83 的**跨进程**能力 ⇒ 部署必须用 SQLite(同机) 或 Redis(推荐)，默认 `MemoryStore` 不满足（它在单进程内仍可用，只是跨不了进程）。SQLite 多模块并发写有数据库级锁（WAL 缓解）；alfred 那种多后台模块高并发写，**Redis 最稳**。

> 相关：当前 `Milkie` 构造对 stateStore 有隐式 `?? new MemoryStore()` fallback，生产忘传会静默拿到 mem——这个陷阱由 **#99** 单独修（改必传），不在本设计范围。

---

## 5. 详细设计

### 5.1 存储：key 契约（对 Python 公开，需版本化）

```
context:${contextId}:var:${name}   →   JSON.stringify(value)
```

- `name`：变量名（如 `workspace_instructions`）。约定 `name` 不含 `:`（避免歧义）。
- `value`：`JSONValue`（string / number / boolean / null / array / object）。存储时 `JSON.stringify`，读取时 `JSON.parse`。
- 选 `:var:` 段，与既有 `checkpoint-run` / `interrupt` / `children` 不冲突。
- **契约版本**：本文档 §5.1 即契约 v1。任何对 key 形态 / 编码的更改都要 bump 版本并通知 alfred 侧。

### 5.2 `IStateStore` 新增 `list(prefix)`（公共接口变更）

`listContextVars` 和"invoke 时读该 context 全部 vars"都要枚举，而现接口无 list。新增：

```ts
// src/types/store.ts
export interface IStateStore {
  set(key: string, value: unknown, ttl?: number): Promise<void>
  get(key: string): Promise<unknown>
  delete(key: string): Promise<void>
  exists(key: string): Promise<boolean>
  list(prefix: string): Promise<Array<{ key: string; value: unknown }>>   // ← 新增
}
```

三个实现：
- **MemoryStore**：遍历内部 Map，过滤 `key.startsWith(prefix)`，跳过已过期项。
- **SQLiteStore**：`SELECT key, value FROM kv WHERE key LIKE ? AND (expires IS NULL OR expires > ?)`，参数 `prefix || '%'`。
- **RedisStore**：`SCAN MATCH prefix*`（**不用 `KEYS`**，避免大库阻塞）+ 批量 `MGET`。

> ⚠️ 这是公共接口变更——所以本 issue 走 repo design doc 流程（而非 #81/#82 的 issue inline 提案）。

### 5.3 读取与渲染链路

谁读、何时读：

```
Milkie.invoke(request)
  │
  ├─ 1. 读持久变量：stateStore.list(`context:${contextId}:var:`)
  │        → sessionVars: Record<string, JSONValue>   （本次 invoke 的【快照】）
  │
  ├─ 2. 取本轮变量：request.variables                  （#82，turn 级）
  │
  ├─ 3. 传给 AgentRuntime：{ sessionVariables: sessionVars, variables: request.variables }
  │
  └─ 4. AgentRuntime 在 run() 开头：
           setSessionContext(sessionVars)   → session-context region
           setTurnContext(request.variables) → turn-context region（#82 已有）
```

**隔离语义（重要）**：第 1 步在 invoke **入口处一次性快照读取**。run 进行中，外部进程再 `setContextVar` **不影响本轮**，要到**下一次 invoke** 才生效。
- 好处：一轮之内 agent 看到的变量是稳定的，不会中途跳变；也匹配 alfred 验收"**下次 invoke 可见**"。

### 5.4 session-context region 与 #82 turn-context 的叠加（用实例讲透）

两个 region 各渲染一份变量。问题：如果 `sessionVars` 和本轮 `variables` 有**同名 key**，agent 会看到两份，可能矛盾。

**实例**。假设：
- store 里持久变量：`workspace_instructions = "用中文"`，`session_id = "s-9"`
- 本轮 invoke 传：`current_time = "10:00"`（注意：和持久变量**没有**重名）

渲染出的 messages：

```
[session-context]  --- Session Context ---
                   session_id: s-9
                   workspace_instructions: 用中文
[turn-context]     --- Turn Context ---
                   current_time: 10:00
```

→ 干净，无冲突。**这是 alfred 的典型情况**（session 放持久指令，turn 放每轮时间戳，key 天然不重叠）。

**冲突的边缘情况**。假设本轮 invoke 又传了 `workspace_instructions = "改用英文"`（和持久变量重名）。两种处理：

- **(A) turn 覆盖 session（推荐）**：渲染 session-context 时**剔除**被本轮覆盖的 key：
  ```
  [session-context]  --- Session Context ---
                     session_id: s-9                    ← workspace_instructions 被剔除
  [turn-context]     --- Turn Context ---
                     workspace_instructions: 改用英文    ← 本轮值生效
                     current_time: 10:00
  ```
  代价：这一轮 session-context 内容变了（少了一行）→ 它当轮 cache 失效。但**这是低频事件**（多数轮无重名），平时 session-context 稳定命中。

- **(B) 不去重**：两个 region 各显示一份，prompt 里同时出现两个 `workspace_instructions`，靠约定"turn 优先"。cache 更稳，但 agent 可能困惑。

→ **✅ 已定 (A)**（O2）：turn 覆盖 session，渲染 session-context 时剔除被本轮覆盖的 key。

### 5.5 并发与隔离

- **不同 key 并发写**：per-key 存储天然无竞态（heartbeat 写 `current_time`、cron 写 `workspace_instructions` 互不干扰）。这正是不选"单 blob"的回报。
- **同一 key 并发写**：store 层 last-write-wins，可接受，不加锁。
- **读写隔离**：见 §5.3，invoke 入口快照。

### 5.6 Milkie API（挂在 Milkie 上，对称 `interrupt`）

```ts
class Milkie {
  getContextVar(contextId: string, name: string): Promise<JSONValue | undefined>
  setContextVar(contextId: string, name: string, value: JSONValue, ttl?: number): Promise<void>
  deleteContextVar(contextId: string, name: string): Promise<void>
  listContextVars(contextId: string): Promise<Record<string, JSONValue>>
}
```

内部都是对 `this.stateStore` 的 get/set/delete/list 加上 key 前缀拼装 + JSON 编解码。

### 5.7 生命周期

- 显式 `deleteContextVar` 删除单个；`setContextVar(..., ttl)` 设过期（复用 `IStateStore.set` 的 ttl）。
- **不做自动 GC**：避免误删 alfred 还在用的 `session_created_at` 这类长期变量。
- 会话整体清理由 alfred 侧负责（它知道会话何时结束）。

---

## 6. 跨进程拓扑（部署视角）

```
┌─────────────── alfred 进程（Python）───────────────┐
│  web "/chat"  ─────────────► HTTP ─► Node sidecar  │  ← 对话走 invoke（#86 门面，可选）
│  heartbeat / cron / workflow ─────► 直连 store      │  ← 后台变量走 §5.1 契约，绕过 Node
└────────────────────────────────────────────────────┘
                                         │
                          ┌──────────────┴───────────────┐
                          │   Redis / SQLite（共享后端）   │
                          └──────────────────────────────┘
                                         ▲
┌─────────────── Node 进程（milkie）──────────────────┐  │
│  Milkie.invoke  ──► 入口读 context:{id}:var:* ──────┼──┘
│  Milkie.setContextVar / getContextVar ─────────────┘
└────────────────────────────────────────────────────┘
```

关键：**对话路径**（invoke）和**后台变量路径**（直连 store）解耦。后台模块不必依赖 Node 在线。

---

## 7. 与相邻概念的边界

| 概念 | 谁写 | 住哪 | agent 怎么看到 | 与 #83 关系 |
|---|---|---|---|---|
| **WorkingMemory** | agent 自己（cognitive tools） | checkpoint（event 投影） | wm region（system 段） | 正交，不复用 |
| **#82 turn variables** | 调用方每轮 invoke 传 | 不落盘（volatile） | turn-context region | 共享数据形态；turn 覆盖 session |
| **#83 context vars** | 外部进程 / SDK | 共享 store（持久） | session-context region | 本设计 |
| **checkpoint** | runtime（event 投影） | event log + 路由指针 | 恢复 fsm/wm/regions | 不混入；vars 独立存 |

**为什么不复用 WorkingMemory？** WM 是 agent 的"思考草稿"，住 checkpoint（只读投影），外部进程根本写不进去；而 context var 是**环境写、agent 只读**。语义和归属都不同，强行合并会糊。

---

## 8. 验收

1. 外部（Python via 直连 store，或 SDK）对一个活跃 contextId 写入变量；下一次 invoke 时该变量出现在 agent 收到的 prompt 里。
2. 读回已写入的变量值正确（含 list 枚举全部）。
3. 部署在 **Redis / SQLite** 后端下，Node 进程与另一进程对同一 contextId 的读写互相可见。
4. 仅 session vars 变化时，**system + history 前缀保持稳定**（cacheBreakpoint 在 history 末），只重算 session-context 及其后的小尾巴——越来越长的 history cache 不被破坏。

---

## 9. 开放点决议（已全部拍板，2026-06-01）

### O1 ⭐ session-context 放 **message 段** 还是 **system 段尾部**？

> **✅ 已定（2026-06-01，经 history-cache 论证修正）：(a) message 段，放在 `history` 之后、`turn-context` 之前。**
>
> **修正理由（关键）**：session vars 并非全程不变——`workspace_instructions` 明确"可能本轮变化"。而 `system` 段在 `messages`（history）**之前**：若把会变的 session-context 放进 system 段，它一旦变化就会击穿身后**整个 messages（含越来越长的 history）**的 prefix cache，而 history 是 cache 的最大受益者。放在 history **之后**，变化点只波及 `session-context + turn-context + current-turn` 这一小截尾巴，**history 大 cache 得以保住**。
>
> **cache breakpoint**：设在 **history 末**（让 `system + history` 进 cache），取代/补充现有的 `system-end` 断点。这是本设计对 cache 机制的主要改动，需在实现中处理。
>
> **代价（接受）**：`workspace_instructions` 这类系统指令型变量出现在 user message 段，语义稍弱。cache 正确性优先。

这决定 workspace_instructions 这类"系统指令型"变量的语义与 cache 行为。

> 注：曾倾向 (b) system 段（语义更正），但被"system 段在 history 之前、一变就废 history cache"的论证否决。真正全程不变的事实（如 `session_id`）放 system 段本无害，但为统一与安全，所有 session vars 一律放 history 之后——不变的变量放这里也照样命中 cache，没有损失。

- **(a) message 段**（和 #82 turn-context 一样，放 history 之后）
  - ✅ 完全不碰 system 前缀，cache 最稳
  - ✅ 实现最简单（复用 #82 机制，只改 interTurn/stability）
  - ⚠️ `workspace_instructions` 作为"系统指令"却出现在 user message 里，语义偏弱
- **(b) system 段尾部 + cacheBreakpoint**（放在 wm 之后、footer 之前）
  - ✅ 语义正确：系统指令就该在 system prompt 里
  - ✅ session-stable，变化时只重算 system 尾部 + 之后，不影响 header/skills 前缀
  - ⚠️ session vars 一变就击穿 system 尾部 cache（但低频，可控）
  - ⚠️ 实现要动 system 段的 cacheBreakpoint 逻辑，复杂一点

> 我的倾向：**(b)**，因为这些变量本质是"系统指令/会话事实"，放 system 段语义最正，且 session-stable + cacheBreakpoint 已是 region substrate 现成能力。但 (a) 更省事、cache 最稳。**这条你拿主意。**

### O2 同名 key 的叠加策略 — **✅ (A) turn 覆盖 session**
渲染 session-context 时剔除被本轮 `variables` 覆盖的 key；turn-context 渲染本轮值。多数轮无重名，session-context 稳定命中 cache。

### O3 `IStateStore.list` 三实现 — **✅ 全做（Memory + SQLite + Redis）**
Memory 单测要用；SQLite/Redis 是真正的跨进程后端。MemoryStore 的 list 仍实现（仅不满足跨进程）。

### O4 `listContextVars` 是否 P0 — **✅ 是**
alfred 要"枚举一个会话有哪些变量"，P0。

### O5 value 类型 — **✅ `JSONValue`**
与 #82 一致，够用；不支持非 JSON（二进制）。

---

## 10. 实现计划（评审通过后，TDD）

1. `IStateStore.list(prefix)` + 三实现（Memory/SQLite/Redis）+ 单测（红：list 不存在 → 绿）。
2. `makeSessionContextRegion`（message 段，`section: 'session-context'`，`history` 之后、`turn-context` 之前；`session-stable` / `session-persistent`）+ section 注册 + region 层单测。
3. **cache breakpoint 改到 history 末**（让 `system + history` 进 cache，session-context 变化只重算其后）+ 单测：仅 session vars 变、history 不变时 history 段渲染逐字节稳定。
4. `AgentRuntime.setSessionContext(sessionVars)` + run 接线 + 端到端单测（注入可见）。
5. `Milkie` 4 个 API（get/set/delete/list ContextVar）+ key 编解码 + 单测。
6. `Milkie.invoke` 入口 `list('context:{id}:var:')` 快照读取 + 叠加去重（O2：turn 覆盖 session）+ 端到端。
7. 契约文档定稿（§5.1）+ 验收回归（含 **SQLite 后端跨"实例"读写**测试：一个 Milkie 写、另一个 Milkie 读同一 contextId 变量）。
