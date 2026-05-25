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
| Skill 寿命管理 | `skill_request` 接 `scope: 'turn' \| 'session'` 声明寿命；**不存在 `skill_release` 工具**；turn-end 引擎按 interTurn scope 自动结晶（详见 §4.3） |
| Tool 结果策略 | `ToolResultStrategy` 三正交轴：`shape`（怎么变内容）/ `visibility`（何时进 LLM context）/ `target`（存哪里）；安全默认 `truncate(4000) + inline + scratchpad`（详见 §4.4） |

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
  | 'turn-local'              // turn 结束自动 release（scratchpad / scope='turn' skill / current-turn）
  | { kind: 'ttl'; deadline: number }  // epoch ms 到了 release
  | 'summarize-on-overflow'   // budget 超额时压缩而非丢弃
  | 'promote-to-wm'           // turn-end 时把该 region 转成 wm region（interTurn='session-persistent'）

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

### 4.3 Skill 寿命模型（lifetime declaration + boundary crystallization）

工具 surface 只有 `skill_request`，**没有 `skill_release`**。agent 不承担"我什么时候用完该卸载"的资源管理责任——那是 substrate 的事。

```typescript
// 系统级工具签名
skill_request({
  name:  string,
  scope?: 'turn' | 'session'   // 默认 'turn'
})
```

寿命语义：

| `scope` | 对应 InterTurnScope | 装配 section | 何时清掉 |
|---|---|---|---|
| `'turn'`（默认） | `'turn-local'` | `session-skills` | turn-end 引擎自动 |
| `'session'` | `'session-persistent'` | `persistent-skills` | 永不自动清（除非 budget 超额 evict） |

**为什么没有 `skill_release`**：

| 维度 | `skill_release` 模型 | 寿命声明 + 自动结晶 |
|---|---|---|
| Agent 心智负担 | 高（要记得 release） | 低（声明完就放下）|
| 资源泄漏风险 | 高（忘 release 就泄漏） | 零（边界引擎兜底）|
| LLM 反复 request+release 抖动 | 可能 | 不存在 |
| 跨轮"知识沉淀" | 无机制 | 有（`promote-to-wm`）|
| 工具表面积 | 2 个 | 1 个 |

唯一让步：agent 不能在一个 turn 内"我现在不需要 verifier 了赶紧释放节省 token"。该场景实际罕见（一个 turn 通常聚焦一个问题），不值得为它引入完整 release 通道。

**端到端路径（load `scope='session'` 为例）**：

```
agent LLM 输出 tool_call: skill_request({ name: 'verifier', scope: 'session' })
        │
        ▼
src/tools/system.ts 的 skill_request handler:
  ctx.queueSkillLoad?.(name, scope)
  return { loaded: name, scope, status: 'pending_next_epoch' }
        │
        ▼
ToolContext.queueSkillLoad → AgentRuntime:
  pendingSets.push(makeSkillRegion(name, scope))
        │
        ▼
下一次 FSM step 边界 → runIntraTurnEngine:
  for (const r of pendingSets) regions.set(r.id, r)
        │
        ▼
下次 assemble 时 persistent-skills section 包含 verifier
        │
        ▼
turn-end → runInterTurnEngine 的 crystallization:
  region.interTurn === 'session-persistent' → keep
  （若是 scope='turn' 则 'turn-local' → drop）
```

`makeSkillRegion` 把寿命声明翻译为 Region：

```typescript
function makeSkillRegion(name: string, scope: 'turn' | 'session'): Region {
  return {
    id:        `skill:${normalize(name)}`,
    target:    'system',
    section:   scope === 'session' ? 'persistent-skills' : 'session-skills',
    intraTurn: 'turn-persistent',
    interTurn: scope === 'session' ? 'session-persistent' : 'turn-local',
    stability: scope === 'session' ? 'session-stable' : 'turn-stable',
    content:   loadSkillContent(name),
    format:    c => `--- Skill: ${name} ---\n${c}\n`,
  }
}
```

注意：同一个 skill 被多次 `skill_request` 用不同 scope 调用 = 同一个 region id 被 `set` 覆盖，最后一次 scope 胜出。这是 `Map.set` 的自然 upsert 语义，不需要特殊处理。

### 4.4 Tool result strategy（shape / visibility / target 三正交轴）

不同工具的返回值形态完全不同：`list_dir` 返回 200 字符目录列表，`download_file` 返回 2MB 二进制，`grep` 返回带上下文的 50 个匹配。一刀切的处理策略必然在某一端失败——`verbatim` 让大返回值吃光 budget，`truncate` 让小返回值丢精度。

策略是 **工具自身的属性**，在 `ToolDefinition` 上声明；AgentRuntime 在创建 scratchpad region 之前应用。

```typescript
type Shape =
  | 'verbatim'                                              // 原样存
  | { kind: 'truncate'; maxChars: number; tailHint?: boolean }  // 前 N 字符
  | { kind: 'tail';     maxChars: number }                  // 后 N 字符（日志类）
  | { kind: 'summarize' }                                   // LLM 总结（贵；Phase 2+）
  | { kind: 'extract';  jsonPath: string }                  // JSON 字段提取
  | { kind: 'transform'; fn: (raw: unknown) => unknown }    // 自定义

type Visibility =
  | 'inline'                                                // 默认：region 进下次装配
  | 'stored-only'                                           // region 存在；不进装配；agent 用 context_fetch 拉
  | { kind: 'first-call-then-reference' }                   // 第一次 inline；之后转 stored-only

type Target = 'scratchpad' | 'wm' | 'discard'

interface ToolResultStrategy {
  shape:      Shape
  visibility: Visibility
  target:     Target
  onError?:   Shape    // 失败时另用 shape，默认 'verbatim'（错误信息要完整给 agent）
}

interface ToolDefinition {
  name:           string
  description:    string
  inputSchema:    JSONSchema
  handler:        (input: unknown, ctx: ToolContext) => Promise<unknown>
  resultStrategy?: ToolResultStrategy   // 缺省 = 安全默认（见下）
}
```

**为什么三轴正交而不是单维 enum**：

- `shape` 是"怎么把 raw 变成存储内容"——纯文本变换
- `visibility` 是"这个内容如何/何时进 LLM context"——和 storage 独立
- `target` 是"region 物理放哪个 section"——决定 lifecycle

之前一稿把 `tail(500)` 当作"shape + visibility 复合"，结果是 `download_file` 这种"想存但不想 inline"的场景被硬塞进 shape 维度。三轴拆开后每个轴独立选择，组合自然覆盖所有场景。

**安全默认值**：

```typescript
const DEFAULT_TOOL_RESULT_STRATEGY: ToolResultStrategy = {
  shape:      { kind: 'truncate', maxChars: 4000 },   // ~1K tokens 上限
  visibility: 'inline',
  target:     'scratchpad',
  onError:    'verbatim',
}
```

默认是 `truncate(4000)` 而**不是** `'verbatim'`——大多数 agent 配置事故来自"忘记设上限"。让 `'verbatim'` 成为工具作者主动 opt-in 表达"我保证我不会很大"，比默认放任安全得多。

**示例**：

```typescript
// 小返回值，verbatim
{ name: 'list_dir',
  resultStrategy: { shape: 'verbatim', visibility: 'inline', target: 'scratchpad' } }

// 大日志，只看尾部
{ name: 'tail_log',
  resultStrategy: { shape: { kind: 'tail', maxChars: 500 }, visibility: 'inline', target: 'scratchpad' } }

// 反思工具，结果直接学到 wm 长期保留
{ name: 'self_critique',
  resultStrategy: { shape: 'verbatim', visibility: 'inline', target: 'wm' } }

// 大文件下载，存但不 inline（agent 想看用 context_fetch）
{ name: 'download_file',
  resultStrategy: { shape: 'verbatim', visibility: 'stored-only', target: 'scratchpad' } }

// 纯 side-effect 工具
{ name: 'audit_log',
  resultStrategy: { shape: 'verbatim', visibility: 'inline', target: 'discard' } }

// 结构化提取 + 失败时给完整 error
{ name: 'http_get',
  resultStrategy: {
    shape:      { kind: 'extract', jsonPath: 'data.items[*].{id,title}' },
    visibility: 'inline',
    target:     'scratchpad',
    onError:    'verbatim',
  } }
```

**`stored-only` 的反向通道**：系统级工具 `context_fetch({ regionId })` 把 stored-only region 临时提升成下次 LLM 调用 inline（之后回到 stored-only）。

**未来扩展（不在 Phase 1）**：
- `aggregation`: 同工具多次调用归并到同一个 region
- agent-per-call override: 让 agent 在 tool_use input 里临时改 shape
- `{ kind: 'fit-budget'; maxFraction: 0.2 }`: shape 根据 context 剩余 budget 动态决定截断长度
- `'first-call-then-reference'` visibility 的 Phase 1 实现细节（需要 region 上的"消费计数"）

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

### 7.3 Inter-turn 引擎（invoke 结束 / 下次 invoke 开始时跑）—— Crystallization

turn 边界是 **crystallization（结晶）** 决策点：每个 region 走自己的 `interTurn` 规则决定留不留、留什么。crystallization 不是机械清空，是 region 创建时的寿命声明在这里被兑现。

```typescript
function runInterTurnEngine(regions: ContextRegions, ctx: {
  boundary:    'turn-end' | 'turn-start'
  currentTurn?: { userInput: string }
  now:         number    // IOPort.now()，replay 可决定性
  subAgentId?: string
}): { events: BoundaryDelta[], crystallization?: CrystallizationSummary } {
  const events: BoundaryDelta[] = []
  
  if (ctx.boundary === 'turn-end') {
    const summary: CrystallizationSummary = {
      kept: [], dropped: [], promoted: [], archivedPair: undefined,
    }
    
    // ──── Step 1: 提取最终答案，归档为 history pair region ────
    //
    // 规则：倒序扫 scratchpad section，找第一个满足
    //   content.role === 'assistant' && 不含 tool_use
    // 的 region —— 它是本轮 LLM 给出的最终文本答案。
    //
    // 把它和 current-turn region 的 user input combine 成一个 history pair region。
    // 所有中间 tool_use / tool result region 在 step 2 被 'turn-local' 规则丢弃。
    //
    // 想让中间步骤的有用信息跨轮 → 工具用 target: 'wm' / 'discard'，
    // 或 agent 在 turn 内主动调 wm_promote 工具，
    // 或 region 创建时 interTurn: 'promote-to-wm'。
    if (ctx.currentTurn) {
      const finalAssistantText = extractFinalAssistantText(regions)
      const turnPairId = `history:turn-${ctx.now}`
      regions.set(turnPairId, {
        target:    'message',
        section:   'history',
        intraTurn: 'turn-persistent',
        interTurn: 'session-persistent',
        stability: 'session-stable',
        content: {
          pair: [
            { role: 'user',      content: ctx.currentTurn.userInput },
            { role: 'assistant', content: finalAssistantText },
          ],
        },
        format: (c: any) => c.pair as Message[],   // 展开成两条 message
      })
      events.push({ kind: 'added', id: turnPairId, reason: 'turn-archived' })
      summary.archivedPair = turnPairId
    }
    
    // ──── Step 2: 按 interTurn 规则逐个结晶 ────
    for (const r of [...regions._allRegions()]) {
      // 跳过刚归档的 pair（它已经是 session-persistent，留下即可）
      if (r.section === 'history' && r.interTurn === 'session-persistent') {
        summary.kept.push(r.id)
        continue
      }
      
      if (r.interTurn === 'session-persistent') {
        summary.kept.push(r.id)
        continue
      }
      
      if (r.interTurn === 'turn-local') {
        // scratchpad / scope='turn' skill / current-turn user input 都属此类
        regions.delete(r.id)
        events.push({ kind: 'removed', id: r.id, reason: 'turn-local-released' })
        summary.dropped.push(r.id)
        continue
      }
      
      if (typeof r.interTurn === 'object' && r.interTurn.kind === 'ttl') {
        if (ctx.now > r.interTurn.deadline) {
          regions.delete(r.id)
          events.push({ kind: 'removed', id: r.id, reason: 'ttl-expired' })
          summary.dropped.push(r.id)
        } else {
          summary.kept.push(r.id)
        }
        continue
      }
      
      if (r.interTurn === 'promote-to-wm') {
        // 转成 wm region（target='system', section='wm', interTurn='session-persistent'）
        const promotedId = `wm:${r.id}`
        regions.set(promotedId, {
          ...r,
          id:        promotedId,
          target:    'system',
          section:   'wm',
          interTurn: 'session-persistent',
          stability: 'session-stable',
        })
        regions.delete(r.id)
        events.push({ kind: 'added',   id: promotedId, reason: 'promoted-to-wm' })
        events.push({ kind: 'removed', id: r.id,       reason: 'promoted-source-removed' })
        summary.promoted.push({ from: r.id, to: promotedId })
        continue
      }
      
      if (r.interTurn === 'summarize-on-overflow') {
        // 检查整体 budget；超额时调用 summarize（Phase 2+）
        // Phase 1: 视为 'session-persistent' 不动
        summary.kept.push(r.id)
        continue
      }
    }
    
    return { events, crystallization: summary }
  }
  
  if (ctx.boundary === 'turn-start') {
    // 1. 从 checkpoint 还原的 session-persistent regions 已经在 regions Map 里
    // 2. 把本轮 user input 注册成 current-turn region (interTurn='turn-local')
    //    （由 AgentRuntime.invoke 在调用本函数后做，不在引擎内）
  }
  
  return { events }
}

interface CrystallizationSummary {
  kept:         string[]
  dropped:      string[]
  promoted:     Array<{ from: string; to: string }>
  archivedPair: string | undefined
}

// 辅助函数
function extractFinalAssistantText(regions: ContextRegions): string {
  const scratch = [...regions._allRegions()]
    .filter(r => r.section === 'scratchpad')
    .filter(r => (r.content as any)?.role === 'assistant')
    .sort((a, b) => b.createdAt - a.createdAt)
  for (const r of scratch) {
    const msg = r.content as any
    if (!msg.toolUse && msg.text) return msg.text
  }
  return ''
}
```

**关键设计要点**：

- **agent 不需要在 turn 内显式 release**——寿命已在 region 创建时声明
- scratchpad 中间步骤（tool_use / tool result）默认丢弃（属 `turn-local`）
- 想保留中间信息有 **三条声明式通道**（不靠 turn-end 反思决定）：
  1. 工具的 `resultStrategy.target = 'wm'`（结果直接进 wm，永远保留）
  2. 工具的 `resultStrategy.target = 'scratchpad'` + `interTurn = 'promote-to-wm'`（turn-end 自动升级）
  3. agent 调系统工具 `wm_promote({ regionId })` 把 region 的 `interTurn` 改成 `'promote-to-wm'`（agent 反思式标记）
- crystallization 是 **可重放的纯函数**——同 regions Map + 同 ctx → 同 delta + 同 after-state
- `ctx.now` 走 IOPort（和 Phase 4 决定性 invariant 一致）

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
  // 仅 boundary='turn-end' 时存在
  crystallization?: {
    kept:         string[]
    dropped:      string[]
    promoted:     Array<{ from: string; to: string }>
    archivedPair: string | undefined
  }
}

// 工具响应（在既有 tool.responded 上扩展 appliedStrategy 字段）
interface ToolRespondedPayload {
  // 既有字段（toolCallId / toolName / result 等）...
  appliedStrategy?: {
    shapeKind:      'verbatim' | 'truncate' | 'tail' | 'summarize' | 'extract' | 'transform'
    visibility:     'inline' | 'stored-only' | 'first-call-then-reference'
    target:         'scratchpad' | 'wm' | 'discard'
    originalBytes:  number   // raw 返回值长度
    storedBytes:    number   // 应用策略后实际入 region 的长度（discard 时 = 0）
    onErrorPath?:   boolean  // true = handler 抛错走了 onError 策略
  }
}
```

trace inspect 直接能显示"`download_file` 返回 2MB，按 `tail(500)` 截到 500 字节存进 scratchpad"——agent 设计者立刻能发现"我应该改 `stored-only`"。


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

- **UI bug**：`loadedSkills` 字段不存在 → 现在用 region 模型，UI 应该 listen `region.added`/`region.removed`（section='session-skills' 或 'persistent-skills'），而不是 diff `loadedSkills` 数组
- **skill 永不卸载**：example agent 永远 load 不卸 → 重构后**不存在 `skill_release`**（见 §4.3）；agent 用 `skill_request({ scope: 'turn' })` 让 substrate 自动在 turn-end 结晶清理；example 升级后演示的是完整 *load → use → turn-end auto-crystallize* 循环（verifier 在某轮 load 后 turn-end 自动释放，下轮 LLM 重新看不到——符合渐进式披露的本意）
- **scratchpad/history 不分**：example 跨轮跑时 history 累积本轮 noise → 重构后 scratchpad 走 `'turn-local'` interTurn，turn-end 自动清；跨轮 history 只保留 (user, finalAssistant) pair
- **tool 结果无上限**：example 的 `read_file` / `grep` 现在都 verbatim 入 history → 重构后受 `ToolResultStrategy` 管控（example agent.md 可以选择对 `read_file` 用 `truncate(2000)` 限上限，对 `grep` 保持 `verbatim`，对未来"下载附件"类工具用 `stored-only`）

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

**Region 层**：
- **RAG region**：embedding 召回结果作为 turn-local region；需要先有 embedding 基础设施
- **Cost-aware region**：实时告知 agent 剩余预算；需要 token counter 接 cache 数据
- **Time-aware region**：时间戳 / 日期 region；one-shot scope
- **Cross-agent region inheritance**：sub-agent 选择性继承父 agent 的某些 region；需要扩 sub-agent spawn 接口
- **Region templating**：region 内容是 function-of-state（"当前是 N 步，已用 M tokens"）；扩 format 函数签名

**Tool result strategy 层**：
- **`aggregation`**：同工具多次调用归并到同一个 region（trade-off：破坏"每次 LLM 调用 region 状态独立 snapshot"性质，且引发"同 region 多次 mutate 怎么记录到 trace"——需要先想清楚）
- **agent-per-call override**：让 agent 在 tool_use input 里临时改 shape；增加 LLM 误用面，目前留给工具作者 owns
- **`{ kind: 'fit-budget'; maxFraction: 0.2 }`**：shape 根据 context 剩余 budget 动态决定截断长度；需要装配层 expose budget
- **`'first-call-then-reference'` visibility 完整实现**：需要 region 上的"消费计数"和"上次被 LLM 看见时间"

**Cache 层**：
- **Multi-provider cache breakpoints**：今天只翻译 Anthropic 的 `cache_control`；OpenAI 自动 prefix cache 不需要 breakpoint
- **Adaptive cache-cut placement**：根据实际 cache hit rate 自动调整 SECTION_SCHEMA breakpoint 位置

**Lifecycle 层**：
- **`summarize-on-overflow` 实际实现**：Phase 1 视为 `session-persistent`；Phase 2 接 LLM 总结 budget 超额时压缩
- **`wm_promote` 系统工具**：让 agent 在 turn 内主动把某个已有 region 的 interTurn 标记为 `'promote-to-wm'`
