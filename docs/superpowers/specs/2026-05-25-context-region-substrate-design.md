---
title: Context Region Substrate — region abstraction + lifecycle policies + cache-aware assembly
status: draft (synthesizing brainstorm; pending user sign-off before plan)
created: 2026-05-25
related:
  - ARCHITECTURE.md#agent-runtime
  - docs/superpowers/specs/2026-05-24-agent-docs-qa-example-design.md
  - docs/stories/s-010-skill-versioned-load-and-ab-experiment.md
---

# Context Region Substrate

## 1. 目标与边界

milkie 现在的 `ContextLayer.buildRequest` 用 5 个硬编码字段（systemPrompt / instructions Map / stateInstructions / history / currentTurn）+ 2 个参数（tools / workingMemory）凑出 `ModelRequest`。这套结构有四个具体缺口，每一个都不是孤立的："分区"、"装配"、"生命周期"、"prefix cache"四件事在今天的代码里要么混着、要么散着、要么干脆不存在。

本设计把 `ContextLayer` 重做成 **region 抽象 + section schema 装配 + lifecycle 策略引擎 + cache-aware 排序** 四件事的统一基底。落地后:

- agent / FSM / 自动策略对 context 的所有 mutation 走 **同一组 CRUD 原语**
- LLM 调用前的装配 = **一个纯函数** 跑一次，无散落
- region 寿命 = **声明式** scope，不靠"agent 显式 release"作唯一手段
- prefix cache 命中率 = **架构保证**，不靠运气

### 范围内

- `ContextLayer` 内部数据结构重写为 `Map<id, Region>`
- 三个 mutation 原语（set / delete / get），废弃 `loadInstructions` / `unloadInstructions` / `appendHistory` / `setCurrentTurn` 等 per-type API
- 一个纯函数 `assemble(regions, scope): ModelRequest`，集中所有装配逻辑
- Section schema 按 stability 排序（cache-aware）
- 两轴 lifecycle 策略：intra-turn scope + inter-turn scope
- 边界引擎: FSM step boundary + turn boundary 各跑一次
- 新事件类型: `region.added` / `region.removed` / `context.boundary.applied`
- **scratchpad region 显式独立**（从 history 拆出）
- Adapter 层支持 cache breakpoint 声明（Anthropic-style）

### 范围外（明确不做）

- 不重写 RecordingIOPort / ReplayingIOPort / CacheIndex / Agent Trace 主体
- 不引入 RAG / embedding region（架构上预留，不实现）
- 不引入 cost-aware / time-aware region（架构上预留）
- 不破坏 Phase 1-4 的 invariant（FSM、replay、interrupt、skill loading 行为对外保持兼容）
- **会破坏现有 replay fixtures 的 byte-identical**（详见 §9 迁移）

## 2. 当前实现的四个具体缺口

| # | 缺口 | 代码位置 | 后果 |
|---|---|---|---|
| 1 | 7 个 region 散落硬编码 | `ContextLayer` 5 个字段 + 2 个参数 | 加新 region 类型要改 ContextLayer 自己 + buildRequest |
| 2 | scratchpad 和 history 混在同一数组 | `ContextLayer.history` + `appendHistory` 在 `AgentRuntime:430,454` 调用 | 跨轮 history 持续累积本轮 ReAct 中间产物；token 浪费 + maxHistory 截断吃掉真实对话 |
| 3 | 没有卸载机制（agent 视角） | `tools/system.ts` 只有 `skill_request`；`AgentRuntime` 只有 `pendingSkills`/`applyPendingSkills` | "渐进式披露"只能 load 不能 release |
| 4 | 装配顺序靠 Map 插入序 + 硬编码段落顺序 | `buildRequest` 的 for/if 串接 | 加新 region 类型必须改 buildRequest；mutation 顺序意外影响装配顺序；cache 命中靠运气 |

四个缺口的**共同根源**：region 不是一等数据抽象。

## 3. 设计决策（已 sign-off through brainstorm）

| 决策 | 选择 |
|---|---|
| Region 存储 | `Map<id, Region>`，唯三个原语：`set / delete / get` |
| Mutation API | 废弃所有 per-type 方法（`loadInstructions` 等），mutation 只能通过 set/delete |
| 装配执行点 | **唯一一处**：LLM 调用前最后一步，纯函数 `assemble(regions, scope)` |
| 装配顺序 | section schema 集中在 assemble 函数；section 之间按 schema；section 内按 ordinal/createdAt |
| Section 数量与排序 | 按 stability 反向排（最稳定在最前）；具体见 §6 |
| Lifecycle | 两轴：`intraTurn` + `interTurn`，各自有有限策略集合 |
| 边界引擎 | 两个：FSM step boundary（intra-turn）+ turn boundary（inter-turn） |
| scratchpad/history 分开 | history = inter-turn (user, finalAssistant) 对；scratchpad = intra-turn ReAct trail |
| Cache breakpoint | Region 可声明 `cacheBreakpoint?: boolean` ；adapter 层翻译为 provider-specific cache_control |
| 失败模式 | 非法 mutation（删除不存在 / 装配时 region 形状不合规）→ throw，不静默 |

## 4. 数据结构

### 4.1 `Region`

```typescript
type RegionTarget = 'system' | 'message' | 'tool'

type SystemSection = 
  | 'header'           // base agent prompt
  | 'persistent-skills' // session 期常驻 skill
  | 'tools-static'      // FSM root 级常驻工具描述（极少）
  | 'session-skills'    // 会话期 agent 主动 load 的 skill
  | 'state'             // 当前 FSM state 的 instructions
  | 'tools-state'       // 当前 state 专属工具描述
  | 'wm'                // working memory snapshot
  | 'footer'            // 输出格式约束 / safety rules

type MessageSection =
  | 'history'           // 跨轮 (user, finalAssistant) 对
  | 'current-turn'      // 本轮 user 输入
  | 'scratchpad'        // 本轮 ReAct trail（assistant tool_use / tool result）

type ToolSection = 'default'

type IntraTurnScope =
  | 'turn-persistent'                              // 默认；从插入到 turn 结束
  | { kind: 'state-scoped'; state: string }        // FSM state 退出时自动 release
  | { kind: 'tool-buffer'; remainingCalls: number } // K 次 LLM 调用后自动 release
  | 'one-shot'                                     // 下一次 LLM 调用消费后自动 release

type InterTurnScope =
  | 'session-persistent'      // 跨轮持久；checkpoint 保存恢复
  | 'turn-local'              // turn 结束自动 release
  | { kind: 'ttl'; deadline: number }  // epoch ms 到了 release
  | 'summarize-on-overflow'   // budget 超额时压缩而非丢弃

interface Region {
  id:         string
  target:    RegionTarget
  section:   SystemSection | MessageSection | ToolSection
  
  // section 内排序（可选；缺省按 createdAt）
  ordinal?:  number
  createdAt: number   // 由 set() 自动填，IOPort.now() 来源（Phase 4 兼容）
  
  // 生命周期声明
  intraTurn: IntraTurnScope
  interTurn: InterTurnScope
  
  // 装配 hints
  stability:       'immutable' | 'session-stable' | 'turn-stable' | 'volatile'
  cacheBreakpoint?: boolean    // adapter 层在此 region 之后注入 cache_control
  
  // 内容 + 渲染
  content: unknown
  format:  (content: unknown) => string | Message | ToolSchema
}
```

### 4.2 `ContextRegions`（重写后的 ContextLayer）

```typescript
export class ContextRegions {
  private readonly regions: Map<string, Region> = new Map()
  private epoch: number = 0
  
  set(id: string, region: Omit<Region, 'id' | 'createdAt'>): void {
    const existing = this.regions.get(id)
    const createdAt = existing?.createdAt ?? this.clock()  // update keeps original createdAt
    this.regions.set(id, { id, createdAt, ...region })
    this.epoch++
  }
  
  delete(id: string): boolean {
    const existed = this.regions.delete(id)
    if (existed) this.epoch++
    return existed
  }
  
  get(id: string): Region | undefined {
    return this.regions.get(id)
  }
  
  getEpoch(): number { return this.epoch }
  
  snapshot(): RegionSnapshot { ... }   // 用于 checkpoint
  restore(snap: RegionSnapshot): void { ... }
  
  // 仅 assemble 用，不公开给其他消费者
  _allRegions(): IterableIterator<Region> { return this.regions.values() }
  
  // 注入时钟（IOPort）
  constructor(private readonly clock: () => number) {}
}
```

**关键设计点**：
- `set` 是 upsert，但保留原 createdAt（避免重复 set 同 id 时排序漂移）
- 时钟通过构造函数注入（保证 Phase 4 byte-identical 不被破坏——`clock` 从 IOPort 来 = 走 RecordingIOPort 的 nondet log）
- `epoch++` 在每次结构变更时递增，trace 可以观察 epoch 进展
- mutation 完全是 CRUD；没有 per-type 方法

## 5. 装配函数

```typescript
const SECTION_SCHEMA: Record<RegionTarget, ReadonlyArray<string>> = {
  system: [
    'header',
    'persistent-skills',
    'tools-static',
    // ─── cache breakpoint candidate: stable cut ───
    'session-skills',
    // ─── cache breakpoint candidate: session cut ───
    'state',
    'tools-state',
    'wm',
    // ─── cache breakpoint candidate: turn cut ───
    'footer',
  ],
  message: [
    'history',
    'current-turn',
    'scratchpad',
  ],
  tool: ['default'],
}

interface AssembleScope {
  currentState:    string
  currentTurnId:   string
  currentEpoch:    number   // 用于 TTL evaluation
  subAgentId?:     string
}

export function assemble(regions: ContextRegions, scope: AssembleScope): ModelRequest {
  // 1. Filter by intra-turn scope
  const filtered = [...regions._allRegions()].filter(r => isActive(r, scope))
  
  // 2. Group by target
  const byTarget = groupBy(filtered, r => r.target)
  
  // 3. For each target, order by section schema; within section by ordinal/createdAt
  const systemBlocks = SECTION_SCHEMA.system.flatMap(sec =>
    (byTarget.system ?? [])
      .filter(r => r.section === sec)
      .sort(bySectionLocalOrder)
      .map(r => r.format(r.content))
  )
  const messages = SECTION_SCHEMA.message.flatMap(sec =>
    (byTarget.message ?? [])
      .filter(r => r.section === sec)
      .sort(bySectionLocalOrder)
      .map(r => r.format(r.content) as Message)
  )
  const tools = (byTarget.tool ?? [])
    .sort(bySectionLocalOrder)
    .map(r => r.format(r.content) as ToolSchema)
  
  return {
    model:    /* from agent config */,
    system:   systemBlocks.join('\n'),
    messages,
    tools:    tools.length > 0 ? tools : undefined,
  }
}

function isActive(region: Region, scope: AssembleScope): boolean {
  if (typeof region.intraTurn === 'object' && region.intraTurn.kind === 'state-scoped') {
    return region.intraTurn.state === scope.currentState
  }
  if (typeof region.interTurn === 'object' && region.interTurn.kind === 'ttl') {
    return scope.currentEpoch <= region.interTurn.deadline
  }
  return true
}

function bySectionLocalOrder(a: Region, b: Region): number {
  if (a.ordinal != null && b.ordinal != null) return a.ordinal - b.ordinal
  return a.createdAt - b.createdAt
}
```

**关键性质**：
- assemble 是**纯函数**：同样的 regions + scope → 同样的 ModelRequest
- 没有任何 mutation；不调用 IOPort；不修改 region 状态
- 测试不需要任何 mock；构造 regions 集合，调用 assemble，断言输出
- 加新 region 类型 = `Region.section` 加一个字面量 + `SECTION_SCHEMA` 加一行 + assemble 不动

## 6. Section schema 的 cache 论证

为什么 `SECTION_SCHEMA.system` 是这个顺序而不是别的？按 stability 反向排，使得**最稳定的内容最先序列化**，prefix cache 才能命中最大区间。

| Section | 典型 stability | 装配位置 | Cache 收益 |
|---|---|---|---|
| header | immutable | 最前 | 永久命中（跨 agent invocation） |
| persistent-skills | session-stable | 早 | 跨轮命中 |
| tools-static | session-stable | 早 | 跨轮命中 |
| **── stable cut breakpoint ──** | | | |
| session-skills | session-stable but mutable | 中早 | 命中直到下次 skill load/release |
| **── session cut breakpoint ──** | | | |
| state | turn-stable, changes on FSM transition | 中 | 命中直到下次 state 转移 |
| tools-state | turn-stable | 中 | 同 state |
| wm | volatile | 中后 | 容易破坏后续 |
| **── turn cut breakpoint ──** | | | |
| footer | immutable but late | 末 | **零 cache 收益**（前面 volatile） |

注意 `footer` 的悖论——它本身不变但因为放最末（routinely positioned after volatile content），cache 收益接近零。这是**有意的权衡**：footer 的"输出格式约束"语义上必须最后出现（让 LLM 看到完所有输入后被这一段引导输出），cache 不能优先。把语义错误地塞进前段会失败的更严重。

`messages` 顺序：
- history 必须在前（chronological prefix）
- current-turn 用户输入接 history（这是新内容的起点）
- scratchpad 在最末（每次 LLM 调用都追加，必然破坏后续 cache，但前缀 = history + current-turn 是稳定可缓存的）

## 7. Lifecycle 策略与边界引擎

### 7.1 边界

```
┌────────── 一次 Milkie.invoke (一个 turn) ──────────┐
│                                                    │
│  ┌── step ──┐ ┌── step ──┐ ┌── step ──┐            │
│  │   LLM    │ │   LLM    │ │   LLM    │            │
│  │   tool   │ │   tool   │ │  final   │            │
│  │   ...    │ │   ...    │ │  text    │            │
│  └────▲─────┘ └────▲─────┘ └──────────┘            │
│       │            │                                │
│   FSM step      FSM step                            │
│   boundary      boundary                            │
│   (intra-turn   (intra-turn                         │
│    engine fires) engine fires)                      │
│                                                     │
└─────────────────────────────────────────────────────┘
                       ▲
                       │
                  turn boundary
                  (inter-turn engine fires
                   at invoke end / next invoke start)
```

### 7.2 Intra-turn 引擎（每次 FSM step 边界跑一次）

```typescript
function runIntraTurnEngine(regions: ContextRegions, scope: AssembleScope, ctx: {
  pendingSets: Region[]      // agent skill_request / tool result 等待应用
  pendingDeletes: string[]   // agent skill_release 等待应用
  stateTransition?: { from: string, to: string }
  llmCallCounter: number     // 本 turn 累计 LLM 调用次数
}): { events: BoundaryDelta[] } {
  const events: BoundaryDelta[] = []
  
  // 1. 应用 pending mutations
  for (const r of ctx.pendingSets) {
    regions.set(r.id, r)
    events.push({ kind: 'added', region: r })
  }
  for (const id of ctx.pendingDeletes) {
    if (regions.delete(id)) events.push({ kind: 'removed', id, reason: 'agent-release' })
  }
  
  // 2. state-scoped: state 转移时卸掉旧 state 专属 region
  if (ctx.stateTransition) {
    for (const r of regions._allRegions()) {
      if (typeof r.intraTurn === 'object' && r.intraTurn.kind === 'state-scoped' 
          && r.intraTurn.state === ctx.stateTransition.from) {
        regions.delete(r.id)
        events.push({ kind: 'removed', id: r.id, reason: 'state-exit' })
      }
    }
  }
  
  // 3. tool-buffer: 每次 LLM 调用 remainingCalls--，到 0 释放
  for (const r of regions._allRegions()) {
    if (typeof r.intraTurn === 'object' && r.intraTurn.kind === 'tool-buffer') {
      const next = { ...r.intraTurn, remainingCalls: r.intraTurn.remainingCalls - 1 }
      if (next.remainingCalls <= 0) {
        regions.delete(r.id)
        events.push({ kind: 'removed', id: r.id, reason: 'tool-buffer-expired' })
      } else {
        regions.set(r.id, { ...r, intraTurn: next })
      }
    }
  }
  
  // 4. one-shot: 上次 LLM 调用消费过的 → 释放
  // (需要 LLM 调用层标记 "consumed" 才能精确——简化版按调用次数判断)
  
  // 5. TTL: deadline 过了
  for (const r of regions._allRegions()) {
    if (typeof r.interTurn === 'object' && r.interTurn.kind === 'ttl') {
      if (scope.currentEpoch > r.interTurn.deadline) {
        regions.delete(r.id)
        events.push({ kind: 'removed', id: r.id, reason: 'ttl-expired' })
      }
    }
  }
  
  return { events }
}
```

### 7.3 Inter-turn 引擎（invoke 结束 / 下次 invoke 开始时跑）

```typescript
function runInterTurnEngine(regions: ContextRegions, ctx: {
  boundary: 'turn-end' | 'turn-start'
  currentTurn?: { userInput: string, finalAssistantContent: MessageContent[] }
  subAgentId?: string
}): { events: BoundaryDelta[] } {
  const events: BoundaryDelta[] = []
  
  if (ctx.boundary === 'turn-end') {
    // 1. 把 scratchpad 的最终答案提取，连同 user input 一对作为 (user, assistant) 追加进 history
    if (ctx.currentTurn) {
      const turnPairId = `history:turn-${uuid()}`
      regions.set(turnPairId, makeUserAssistantPair(ctx.currentTurn))
      events.push({ kind: 'added', id: turnPairId, reason: 'turn-archived' })
    }
    
    // 2. 释放所有 turn-local region（scratchpad 全部 / turn-local skill / 任何 inter='turn-local'）
    for (const r of regions._allRegions()) {
      if (r.interTurn === 'turn-local') {
        regions.delete(r.id)
        events.push({ kind: 'removed', id: r.id, reason: 'turn-local-released' })
      }
    }
    
    // 3. summarize-on-overflow: 如果 history 总 token 超 budget，压缩最旧 N 条
    // (实现细节在 §10)
  }
  
  if (ctx.boundary === 'turn-start') {
    // 1. 从 checkpoint 还原的 session-persistent regions 已经在 regions 里
    // 2. 重新 set current-turn region
    // 3. （首次） 触发 ContextLayer 已 ready
  }
  
  return { events }
}
```

### 7.4 边界引擎的 hook 点（对现有 AgentRuntime 的最小改动）

```typescript
// AgentRuntime.executeFSM 现有结构（伪代码）:
while (true) {
  if (state.type === 'llm') {
    const request = this.context.buildRequest(...)   // 今天
    const response = await this.ioPort.invokeLLM(request)
    this.context.appendHistory({assistant, ...})    // 今天
    if (toolCalls) {
      results = await this.executeTools(toolCalls)
      this.context.appendHistory({tool, ...})       // 今天
      this.applyPendingSkills()                     // 今天
    }
    // state 转移逻辑
  }
}

// 重构后:
while (true) {
  if (state.type === 'llm') {
    const { events } = runIntraTurnEngine(regions, scope, {       // ← 新增
      pendingSets, pendingDeletes, stateTransition, llmCallCounter,
    })
    await emitBoundaryEvent(events)                               // ← 新增
    const request = assemble(regions, scope)                       // ← 替代 buildRequest
    const response = await this.ioPort.invokeLLM(request)
    pendingSets.push(makeScratchpadRegion('assistant', response.content))  // ← 替代 appendHistory
    if (toolCalls) {
      results = await this.executeTools(toolCalls)
      pendingSets.push(makeScratchpadRegion('tool', results))     // ← 替代 appendHistory
      // pendingSkills 已经统一进 pendingSets/pendingDeletes
    }
    // state 转移 → 下一次循环顶部的 runIntraTurnEngine 处理
  }
}

// invoke 入口/出口:
await runInterTurnEngine(regions, { boundary: 'turn-start' })
try {
  await this.executeFSM()
} finally {
  await runInterTurnEngine(regions, { 
    boundary: 'turn-end', 
    currentTurn: { userInput, finalAssistantContent: this.lastAssistantContent } 
  })
}
```

## 8. Trace 可观察性

### 8.1 新事件类型

```typescript
// 单条 region 变更（用于细粒度审计）
interface RegionAddedPayload {
  region: Region
  reason: 'agent-set' | 'fsm-state-enter' | 'turn-start' | 'turn-archive' | string
}
interface RegionRemovedPayload {
  id:     string
  reason: 'agent-release' | 'state-exit' | 'tool-buffer-expired' | 'ttl-expired'
        | 'turn-local-released' | 'budget-evicted' | 'one-shot-consumed'
}

// 边界引擎运行结果（一次边界 fire 的完整 delta + after-snapshot）
interface ContextBoundaryAppliedPayload {
  boundary:  'fsm-step' | 'turn-end' | 'turn-start'
  epoch:     number
  delta: {
    added:   Array<{ id: string, section: string, reason: string }>
    removed: Array<{ id: string, reason: string }>
  }
  activeAfter: Array<{   // 完整快照（便于 trace 跳转到任意边界查"那时 context 长什么样"）
    id: string
    section: string
    stability: string
    tokenEstimate?: number
  }>
}
```

### 8.2 Cache 健康度

LLM 调用响应里如果带 `usage.cache_read_tokens` / `usage.cache_creation_tokens`（Anthropic / OpenAI），写进 `llm.responded` 事件：

```typescript
interface LlmRespondedPayload {
  // 既有字段...
  cacheStats?: {
    readTokens:    number   // 命中段
    creationTokens: number  // 写入段
    totalInputTokens: number
    hitRate: number  // = readTokens / totalInputTokens
  }
}
```

trace inspect / HTML report 可以直接展示"本次 LLM 调用 cache 命中 90%"，agent 设计者立刻能看到哪些操作在烧钱。

### 8.3 与 Phase 4 替代关系

- 之前讨论的 `skill.loaded` / `skill.released` 事件 **作废**——被更通用的 `region.added` / `region.removed` 包含
- skill 只是 `region.added` 中 `region.section === 'session-skills'` 的特例

## 9. 迁移路径

### 9.1 对现有 replay fixture 的影响

**Breaking**: 现有 replay fixtures（`examples/s-005-replay/.milkie/runs/*.jsonl`、`examples/s-002-inspect/.milkie/runs/*.jsonl`、`tests/e2e/s-005-*` 等）的 byte-identical 都会 break:

- 旧 fixture 没有 `region.*` / `context.boundary.applied` 事件 → 重放时事件队列里少东西
- 装配产出的 LLM request 字节级会变（因为 section schema 调整后 system 字符串结构变了）→ 之前录的 LLM 响应 cache 命中失败

**两种迁移策略**：

**A. 全部重新录制 fixture**（推荐）  
跟着 spec 实现一起：删旧 fixture → 跑 record.ts → 提交新 fixture。新 fixture 自然包含 region 事件 + 新装配格式。  
代价：s-005 e2e + s-002 example + 任何依赖现有 JSONL 的测试都得跑一遍 + commit 新 hash。

**B. 兼容性 shim**（不推荐）  
ContextLayer 内部用 region 模型，但 buildRequest 输出的 system 字符串保持现有格式。  
代价：region 抽象的好处只到 ContextLayer 内部，外部仍然看的是 flat string；migration 半拉子，将来还得做一次彻底迁移。

走 A。"未上线无需向后兼容"——和 Phase 4 决策同源。

### 9.2 落地顺序（分阶段）

为了让回滚成本可控、review 颗粒可消化，分 4 个 PR：

| PR | 范围 | 行数估计 | 风险 |
|---|---|---|---|
| **PR-A: Region 数据结构 + Map API + 单元测试** | `src/context/Region.ts`, `src/context/ContextRegions.ts`, `__tests__/ContextRegions.test.ts` | ~500 | 低（独立单元，无消费者）|
| **PR-B: assemble 纯函数 + section schema + 单元测试** | `src/context/assemble.ts`, `src/context/sectionSchema.ts`, `__tests__/assemble.test.ts` | ~400 | 低（仍独立，只消费 PR-A）|
| **PR-C: AgentRuntime 接入 + 边界引擎 + scratchpad 拆出 + replay fixture 重录 + 全 e2e 跑通** | 大头：`src/runtime/AgentRuntime.ts` 改造、`src/context/lifecycleEngine.ts` 新建、所有 fixture 重录、所有 e2e 测试更新 | ~1500-2000 | **高**（接入面广，e2e 多）|
| **PR-D: trace 事件类型 + cache health + adapter cache_control + HTML report 渲染 region 事件** | `src/trace/types.ts`、各 adapter、HTML report | ~600 | 中 |

PR-A / PR-B 可并行做（互不依赖），PR-C 必须等两者都 land，PR-D 等 PR-C。

## 10. 对当前 PR3 (agent-docs-qa example) 的影响

### 10.1 PR3 现状重审

带着这次设计的视角回看 PR3 demo 的几个具体问题：

- **UI bug**：`loadedSkills` 字段不存在 → 现在用 region 模型，UI 应该 listen `region.added`/`region.removed` 而不是 diff `loadedSkills` 数组
- **没 release**：example agent 永远 load 不卸 → 重构后 `skill_release` 工具自然存在，example 可以演示完整 load+release 循环
- **scratchpad/history 不分**：example 跨轮跑时 history 累积本轮 noise → 重构后干净

### 10.2 PR3 的两条路

**路 1: 合 PR3 后再做 substrate（推荐）**

- PR3 当前的实现是基于 milkie 当前 substrate 的诚实最大值。它确实证明了 load 半边 + 实时 UI + 自动 e2e
- README 里补一节"Known substrate gaps"：诚实写明今天 UI 的 loadedSkills 检测不工作 / 没 release / scratchpad mixed in history。指向本 design spec 作为下一步
- 之后做 substrate 重构（4 个 PR），到 PR-C 时更新 PR3 例子的 UI（改成 listen region 事件）+ agent.md（演示 release）

**路 2: 阻塞 PR3 直到 substrate 重构完成**

- 不 merge PR3。先做 4 个 substrate PR，再回头把 PR3 的实现按新 substrate 重写后 merge
- 代价：substrate 是 1500-3000 行工作量，PR3 等几周；期间 substrate 设计可能再变也会影响 PR3 rewrite

**我倾向路 1**：PR3 单独有价值（trace 实时可观察 / 自动 e2e 守住 skill loading load 路径 / 完整 web UI demo template），substrate 重构是独立的 substrate 改造，两者解耦更利于 review 和回滚。merge PR3 + 在 README 写明 substrate gaps + 链接到 spec 是合适的诚实文档化。

### 10.3 PR3 README 应补的诚实段落

```markdown
## Known substrate limitations (addressed in pending design)

This example was built on milkie's current ContextLayer, which has several
architecturally-incomplete behaviors:

- **UI loadedSkills detection is dead code**: the AgentRunStartedPayload
  doesn't actually carry a `loadedSkills` field; the yellow skill-load
  highlight in the trace timeline never fires today.
- **No skill release path**: agent can `skill_request` to load verifier
  but has no way to unload it; verifier stays in system prompt for the
  remainder of the conversation.
- **scratchpad and history conflated**: ReAct loop intermediates
  (assistant tool_use + tool result) are appended to the same `history`
  array as cross-turn (user, finalAssistant) pairs, leading to token
  bloat across turns.

All three are addressed in:
`docs/superpowers/specs/2026-05-25-context-region-substrate-design.md`

After that substrate work lands, this example's frontend + agent.md will
be updated to demonstrate the full progressive disclosure cycle (load +
release) with correct UI observation.
```

## 11. 不变式（测试用）

1. **Assemble 是纯函数**：同样的 regions Map + scope → 同样的 ModelRequest（byte-identical）
2. **Mutation 全局观察**：每次 `set` / `delete` 都对应一个 `region.added` / `region.removed` 事件
3. **边界引擎确定性**：同样的 pending mutations + scope → 同样的边界 delta
4. **Section 排序稳定**：相同 section 内 region createdAt 单调 → 排序稳定（cache-friendly）
5. **state-scoped 自动卸**：state 转移后，旧 state 的 state-scoped region 已不在 active set
6. **turn-local 自动卸**：turn-end 后，所有 interTurn='turn-local' region 已不在 regions Map
7. **TTL 准确**：scope.currentEpoch > region.interTurn.deadline 时，下次 assemble 看不到该 region
8. **Cache breakpoint adapter 翻译**：region.cacheBreakpoint=true → adapter 注入 cache_control
9. **scratchpad/history 隔离**：scratchpad 在 turn-end 后全部清空；history 仅含 (user, finalAssistant) 对
10. **rollback safe**: 任何 mutation 失败时 regions Map 不进入半改状态（用 staging set + commit pattern）

## 12. roadmap.md 影响

完成后:

- TL;DR: "Next big rock: Phase 5 fork/diff/suite" 之前加 "Phase 4.5 ContextRegions substrate landed" 一行
- 新 Completed 段: "Phase 4.5 — ContextRegions substrate"
- Open architectural questions: "Deterministic-flow placement" 重提（region 模型让这个问题变得更具体）
- Cross-cutting: "TrajectoryStore retirement decision" 应在 Phase 4.5 之后做（region 模型让 trajectory 的去留更清晰）

## 13. 范围外 / 未来扩展

- **RAG region**：embedding 召回结果作为 turn-local region；需要先有 embedding 基础设施
- **Cost-aware region**：实时告知 agent 剩余预算；需要 token counter 接 cache 数据
- **Time-aware region**：时间戳 / 日期 region；one-shot scope
- **Cross-agent region inheritance**：sub-agent 选择性继承父 agent 的某些 region；需要扩 sub-agent spawn 接口
- **Region templating**：region 内容是 function-of-state（"当前是 N 步，已用 M tokens"）；扩 format 函数签名
- **Multi-provider cache breakpoints**：今天只翻译 Anthropic 的 cache_control；OpenAI 自动 prefix cache 不需要 breakpoint
