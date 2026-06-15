# #162 Portable Hierarchical Entity Resolver

- **Issue**: #162
- **Branch**: `feat/165-design-portable-hierarchical-entity-resolver`
- **状态**: 设计已定稿（#165 评审通过），为 single source of truth；issue #162 body 保留摘要 + 链接指向本文档。
- **落地范围**: `examples/repair-ticketing` 内实现 + s-011 Path D。本阶段不发布独立 npm 包、不承诺稳定公共 SDK、不做跨仓库维护。

---

## 1. 设计定位

**Hierarchical Entity Resolver（HER）不是 AgentRuntime 的核心能力。**

HER 是一个**应用层工具模块**，其职责是：给定用户的自然语言描述，在一个层级化的参考数据集（如维修工单中的「站点 → 楼宇 → 部门 → 负责人」）中，找到最精确的实体节点并校验选择合法性。

它的正确归属：
- core 暂住 `examples/repair-ticketing/resolver/`，作为该 example 的领域模块。
- 对 milkie 的 `AgentRuntime`、`FSMEngine`、`IOPort` **零依赖**。
- milkie adapter（第 3 节）是它在 FSM 工具 handler 中的薄接线层，不是它的一部分。
- **与 #167 的关系**：#167 计划将 core 抽取为独立可发布包；本阶段 core 住在 example 内，本文档定义的边界（core / adapter / CLI）即为 #167 抽取时的参考切割线。

**为什么不内置进 AgentRuntime？**

AgentRuntime 管理 FSM 状态、LLM 调用和 IOPort 边界。把领域专属的层级数据遍历逻辑内置进来会模糊 runtime 的职责边界，且 HER 的数据加载（CSV/JSON）与 replay 不相关，强行接线反而引入不必要的不确定性记录负担。

---

## 2. 架构边界

```
┌─── examples/repair-ticketing ─────────────────────────────────────────┐
│                                                                         │
│   milkie agent (FSM + slot filling)                                    │
│       │                                                                 │
│       └──► milkie adapter                                               │
│            (src/tools/entity-resolver.ts)                              │
│                 │  reads ctx.workingMemory → assembles context.pinned  │
│                 │  in-process call (no subprocess)                      │
│                 ▼                                                       │
│           EntityResolver (pure TS module)                               │
│           resolver/EntityResolver.ts                                    │
│                 │                                                       │
│                 ├── resolver/schema.json   (hierarchy schema)           │
│                 └── resolver/data.csv      (entity data)                │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  CLI wrapper (scripts/resolver.ts)                              │  │
│   │  thin shell over same EntityResolver                            │  │
│   │  stdin → JSON → EntityResolver → JSON → stdout                 │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│   ↑ 供 AgentScope / Python / 非 TS 框架调用（第 4 节）                  │
└─────────────────────────────────────────────────────────────────────────┘
```

**文件路径一览（统一参考）：**

| 角色 | 路径 |
|---|---|
| Core | `examples/repair-ticketing/resolver/EntityResolver.ts` |
| Schema | `examples/repair-ticketing/resolver/schema.json` |
| Data | `examples/repair-ticketing/resolver/data.csv` |
| Adapter | `examples/repair-ticketing/src/tools/entity-resolver.ts` |
| CLI wrapper | `examples/repair-ticketing/scripts/resolver.ts` |

**两条调用路径，共享同一 core：**

| 路径 | 调用者 | 机制 | 何时选用 |
|---|---|---|---|
| 进程内调用（主路径） | milkie adapter | `import { EntityResolver }` | milkie agent 内所有情况 |
| CLI wrapper | Python/AgentScope 等 | `node scripts/resolver.ts` via stdin/stdout | 非 TS 框架需要集成时 |

---

## 3. 主路径 — 进程内调用

milkie adapter 通过 **ES module import** 直接调用 `EntityResolver`，整个过程在同一个 Node.js 进程内完成。

```ts
// examples/repair-ticketing/src/tools/entity-resolver.ts
import { EntityResolver } from '../resolver/EntityResolver.js'

const resolver = new EntityResolver({ schemaPath, dataPath })

export const lookupEntityTool = {
  name: 'lookup_entity',
  async handler(input: LookupInput, ctx: ToolContext) {
    // 从 workingMemory 读取已固定的祖先层 id，组装 context.pinned（§3.1）
    const pinned = buildPinnedFromWorkingMemory(ctx.workingMemory, input.context.level)
    return resolver.lookup({ ...input, context: { ...input.context, pinned } })
  },
}

export const commitEntityTool = {
  name: 'commit_entity',
  async handler(input: CommitInput, ctx: ToolContext) {
    // 同样从 workingMemory 注入 pinned，不信 LLM 参数里的值
    const pinned = buildPinnedFromWorkingMemory(ctx.workingMemory, input.context.level)
    const result = await resolver.commit({ ...input, context: { ...input.context, pinned } })
    // commit 成功后将本层 resolved.id 写回 workingMemory（§3.1）
    if (result.resolved && !result.validationError) {
      ctx.workingMemory[input.context.level] = result.resolved.id
    }
    return result
  },
}
```

`EntityResolver` 本身是**纯同步数据处理**（索引在构造时建完），handler 的 async 包装仅满足 milkie 工具接口约定。

### 3.1 祖先拓扑 Recall — adapter ↔ workingMemory

解析子层时，`lookup` 和 `commit` 需要 `context.pinned`（已确定的祖先层 id）来做层级过滤。**`pinned` 必须由 adapter 从 `ctx.workingMemory` 组装，不能来自 LLM 的工具调用参数。**

原因（三重约束）：

1. **与 #164 初衷一致**：#164 引入 `ctx.currentTurn` 就是把可信上下文数据与 LLM 生成参数分开；祖先拓扑链是结构化约束，属于可信侧。
2. **§7 `corrected` 语义的前提**：`corrected` 触发于 `selected` 的实际祖先链与 `pinned` 不一致（模糊匹配混入了相邻分支的候选）。若 `pinned` 由 LLM 填，LLM 可伪造祖先约束或故意选错分支，`corrected` 的业务语义失效。
3. **与 #162 body 一致**：#162 明确 adapter 须"reads ctx.currentTurn + workingMemory … writes WM after validated result"。

**协议：**

```
每次 commit_entity 成功（result.resolved 存在，无 validationError）：
  ctx.workingMemory[level] = result.resolved.id

每次 lookup_entity 或 commit_entity 调用前：
  pinned = { 所有在 workingMemory 中比 level 更高层级的 name: id }
  注入到 context.pinned，覆盖（忽略）LLM 参数里的同名键
```

**与 `ctx.currentTurn` 对称：**

| 数据 | 来源 | 作用 |
|---|---|---|
| `ctx.currentTurn` | milkie 注入（用户原文） | 用作 `lookup` 的 `query` 补全 |
| `ctx.workingMemory` | adapter 写入（每次 commit 成功后） | 用作 `context.pinned` 层级过滤 |

resolver core 保持纯函数；所有有状态操作（读/写 workingMemory）集中在 adapter 层。

---

## 4. CLI Wrapper — 非 TS 框架的薄门面

CLI wrapper 是同一个 `EntityResolver` core 的**命令行入口**，不包含任何额外逻辑。

```
node examples/repair-ticketing/scripts/resolver.ts <command>
```

通信协议：**单次 stdin → stdout JSON**（无长连接、无状态），调用方负责进程管理。

```
stdin:  { "op": "lookup", "query": "...", "context": { "level": "...", "pinned": {...} } }
stdout: { "candidates": [...], "options": [...], "suggested": "..." }

stdin:  { "op": "commit", "selected": "...", "context": { "level": "...", "pinned": {...} } }
stdout: { "resolved": {...}, "corrected": "..." }
```

**CLI 不是 milkie 的主路径。** milkie agent 进程内已有 `EntityResolver` 实例，不会通过 spawn 子进程调用 CLI。注意：CLI 自身**不维护跨调用的 workingMemory 状态**（无状态 stdin/stdout），祖先拓扑 recall（§3.1）由调用方负责传入 `context.pinned`。

CLI 仅供以下场景：
- AgentScope / Python-based 框架集成（调用方自行维护 pinned 状态）
- 快速命令行调试：`echo '{"op":"lookup","query":"三楼机房","context":{"level":"department"}}' | node scripts/resolver.ts`

---

## 5. JSON 协议

### 5.1 lookup

输入：

```ts
interface LookupInput {
  op: 'lookup'
  query: string            // 用户自然语言描述（或 LLM 提取的关键词）
  context: {
    level: string          // 当前要解析的层级名（如 "site" / "department" / "assignee"）
    pinned?: Record<string, string>  // 已固定的祖先层 id，如 { site: 'S01', department: 'D03' }
                                     // 由 adapter 从 workingMemory 组装（§3.1），非 LLM 填写
    sessionHint?: string   // 可选：本次会话的额外提示（如工单类型）
  }
}
```

输出：

```ts
interface LookupOutput {
  candidates: Array<{
    id: string
    label: string          // 对用户显示的名称
    path: string[]         // 完整层级路径，如 ["总部", "IT部", "网络组"]
    score: number          // 匹配得分（0–1），供 LLM 参考排序
  }>
  options: string[]        // 精简版候选列表（id），供 LLM function calling 填写 selected
  suggested: string | null // HER 认为最可能的候选 id；null 表示置信度不足
}
```

`candidates`、`options`、`suggested` 三者共同呈现给 LLM，让 LLM 做最终选择。`suggested` 是辅助提示，LLM 不必采纳。

### 5.2 commit

输入：

```ts
interface CommitInput {
  op: 'commit'
  selected: string         // LLM 从 lookup.options 中选出的 id
  context: {
    level: string
    pinned?: Record<string, string>  // 同 lookup，由 adapter 注入（§3.1）
  }
}
```

输出：

```ts
interface CommitOutput {
  resolved: {
    id: string
    label: string
    path: string[]
    meta: Record<string, unknown>  // 该实体在 data.csv 中的全部字段
  }
  corrected?: string   // 见第 7 节；存在时表示 selected 被自动修正为 corrected
  validationError?: string  // selected 不在 options 集合内时返回（LLM 幻觉防护）
}
```

---

## 6. `selected` 来源 — LLM 从 lookup 结果生成

`selected` 由 **LLM** 产生，不由 HER 或 adapter 产生。

流程：

```
1. adapter 从 workingMemory 组装 pinned，调用 lookup({ op:'lookup', query, context:{ level, pinned } })
         ↓
2. lookup 返回 { candidates, options, suggested }
         ↓
3. LLM 收到上述结果作为工具返回值，生成下一步工具调用
         ↓
4. LLM 调用 commit_entity({ op:'commit', selected: "<从 options 中选出的 id>", context:{ level } })
         ↓
5. adapter 注入 pinned（从 workingMemory），调用 resolver.commit(...)
         ↓
6. commit 做硬校验（selected ∈ options），返回 resolved（+ 可能的 corrected）
         ↓
7. adapter 将 resolved.id 写回 workingMemory[level]
```

**LLM 是唯一负责"选择"的角色。** HER 提供候选集合和建议，但不替 LLM 做最终决策（`suggested` 只是参考）。`commit` 的硬校验确保 LLM 不能幻觉出一个不存在的实体 id。

---

## 7. `corrected` 语义

`corrected` 是 `commit` 在以下情况下返回的字段：

> **`selected` 是合法的 id（在 lookup 返回的 `options` 集合内，不触发 `validationError`），但其实际所属祖先链与 `context.pinned` 不一致。**

**与 `validationError` 的区别：**

| 情况 | 触发条件 | 含义 |
|---|---|---|
| `validationError` | `selected ∉ options` | LLM 幻觉：id 根本不在候选集合内 |
| `corrected` | `selected ∈ options`，但祖先层校验失败 | 模糊匹配混入了相邻分支的候选；commit 做硬校验发现冲突 |

**示例（提交 `assignee` 层时）：**

- `context.pinned = { site: 'S01', building: 'B01', department: 'D03' }` — adapter 从 workingMemory 注入的祖先链，表示本工单已沿 S01 → B01 → D03 确定到部门层。
- `lookup_entity` 做模糊搜索时，"小张"命中 E007（D03 下）和 E012（D07 下的同名员工）；两者均进入 `options`。
- LLM 选择了 `selected: 'E012'`（id 在 options 内，不触发 `validationError`）。
- `commit` 做祖先层硬校验：E012 的归属部门为 D07 ≠ pinned.department 'D03' → 冲突。
- 返回 `{ resolved: <E007 的实体>, corrected: 'E007' }`，覆盖 LLM 的错误选择。

语义约定：
- `corrected` **存在** → 调用方应用 `corrected` 覆盖 `selected`，向用户解释"该选择不在当前确定的层级范围内，已自动修正"。
- `corrected` **不存在** → `selected` 通过所有校验，`resolved` 是最终实体。
- `corrected` 不用于纠正 LLM 的"猜测偏差"，只用于执行祖先层级约束（来自 `pinned`）。

**`corrected` 的有效性前提**：`pinned` 必须来自 adapter 的 `workingMemory`（§3.1），而非 LLM 参数。若 `pinned` 由 LLM 填写，LLM 可伪造祖先约束绕过 `corrected` 保护，层级校验的业务语义失效。

---

## 8. `schema.json` 与 `data.csv` 的 column mapping

### 8.1 `schema.json`

定义层级结构和字段映射：

```json
{
  "version": 1,
  "levels": [
    { "name": "site",       "label": "站点",   "idColumn": "site_id",  "labelColumn": "site_name" },
    { "name": "building",   "label": "楼宇",   "idColumn": "bldg_id",  "labelColumn": "bldg_name",  "parentLevel": "site" },
    { "name": "department", "label": "部门",   "idColumn": "dept_id",  "labelColumn": "dept_name",  "parentLevel": "building" },
    { "name": "assignee",   "label": "负责人", "idColumn": "emp_id",   "labelColumn": "emp_name",   "parentLevel": "department" }
  ],
  "searchColumns": ["site_name", "bldg_name", "dept_name", "dept_alias", "emp_name", "emp_alias"],
  "metaColumns": ["emp_email", "emp_phone", "dept_head"]
}
```

字段说明：

| 字段 | 含义 |
|---|---|
| `levels[].name` | 层级逻辑名（协议中 `context.level` 和 `pinned` 键的合法值） |
| `levels[].idColumn` | data.csv 中作为 id 的列名 |
| `levels[].labelColumn` | data.csv 中作为显示名称的列名 |
| `levels[].parentLevel` | 父层级名（用于过滤：指定 department 后只展示该 department 下的 assignee） |
| `searchColumns` | 参与模糊匹配的列名列表 |
| `metaColumns` | 原样透传进 `resolved.meta` 的列名列表 |

### 8.2 `data.csv`

平铺的宽表，包含所有层级的字段，每行代表层级树中的一个叶节点：

```csv
site_id,site_name,bldg_id,bldg_name,dept_id,dept_name,dept_alias,emp_id,emp_name,emp_alias,emp_email,dept_head
S01,总部,B01,主楼,D03,IT网络部,网络组,E007,张伟,小张,zhangwei@corp.com,李明
S01,总部,B01,主楼,D03,IT网络部,网络组,E008,王芳,小王,wangfang@corp.com,李明
...
```

**设计约定：**
- CSV 是"宽表"而非范式化多表：`EntityResolver` 在构造时一次性读入并建立倒排索引，后续 lookup 全走内存，无 I/O。
- schema 解耦了列名变化：换一套 CSV 列名只需更新 `schema.json`，代码不变。
- `idColumn` 的值在对应层级内唯一（跨行可重复，因为宽表会有相同 dept_id 出现在多行）。
- `pinned` 的键必须等于 `levels[].name`（如 `department`，而非 `dept`），否则层级过滤不匹配。

---

## 9. Replay 边界 — milkie adapter 不 spawn CLI

### 9.1 不变式风险

milkie 的 replay 不变式（`ARCHITECTURE.md` Phase 3–4）要求：**所有非确定性操作必须经过 IOPort 记录**，replay 时从 cache 回放。具体地：

- `invokeLLM` → 经 `RecordingIOPort` 记录 `llm.requested` / `llm.responded`
- `invokeTool` → 经 `RecordingIOPort` 记录 `tool.requested` / `tool.responded`
- `now()` / `uuid()` → 经 `RecordingIOPort` 记录 `clock.read` / `uuid.generated`

**如果 milkie adapter 通过 `spawn` 子进程调用 CLI**：
- 子进程的 stdin/stdout I/O 不通过 IOPort → **不被记录**
- `tool.responded` 里的返回值依赖子进程的执行结果，而子进程执行不受 `ReplayingIOPort` 控制
- replay 时该工具调用从 cache 回放（正常），但如果 cache miss，子进程会被重新 spawn，可能返回不同结果（数据变更后）→ **破坏 replay 确定性**

### 9.2 规避方式 — 进程内调用 pure module

规避的核心原则：

> **`EntityResolver` 是纯确定性函数**（给定相同的 schema/data，相同的 query，返回相同的结果）。它不需要 IOPort 记录，因为它没有外部副作用。

因此：

1. milkie adapter 通过 **import** 调用 `EntityResolver`，数据在构造时加载到内存，lookup/commit 是纯内存运算。
2. 工具调用本身（`invokeTool`）仍被 `RecordingIOPort` 记录为 `tool.requested` / `tool.responded`。
3. replay 时 `ReplayingIOPort` 直接从 cache 回放工具的返回值，完全不调用 `EntityResolver`（也不需要）。
4. 如果 data.csv 在两次 run 之间发生变化，replay 仍然安全（cache 回放的是第一次 run 时的返回值，不重新执行）。

**明确禁止的模式：**

```ts
// ❌ 错误：spawn 子进程
const result = await execFile('node', ['scripts/resolver.ts'], { input: JSON.stringify(req) })

// ✅ 正确：进程内调用
const result = await resolver.lookup(req)
```

### 9.3 数据加载的 replay 安全性

`EntityResolver` 在 **milkie 首次构造工具实例时**（agent 启动，非每次工具调用）加载 schema 和 data。

- 加载发生在 `Milkie.invoke()` 调用之前（初始化阶段），不经过 IOPort → 不被记录。
- 这是可接受的：数据加载是**确定性初始化**，不是 run 内的非确定性操作。
- 若需要 replay 时数据版本一致，在 CI 或测试中固定数据文件版本即可（同 agent skill markdown 文件的处理方式）。

---

## 10. NOT do（本阶段明确不做）

| 排除项 | 原因 |
|---|---|
| **DAG / 多级并发解析** | 本阶段只做单条路径的线性层级解析（site → building → department → assignee），不做多路分支 DAG |
| **同一工单多值 / 同级多值** | 每个层级只选一个节点；不支持一张工单同时分派给两个部门 |
| **训练式 NER** | lookup 走确定性字符串匹配 + 评分，不引入模型推理，不做实体标注训练数据生成 |
| **tool-bearing skill** | HER 不以 milkie skill（.md 技能文件）形式暴露工具；工具注册在 example 的 agent setup 代码里 |
| **独立 npm 包发布** | 本阶段边界：example 内实现，不建立独立 package + 版本管理（由 #167 决策） |
| **跨仓库维护承诺** | 代码住在 monastery repo 的 example 目录，未来抽取复用以本文档边界为参考 |
| **LLM 驱动的 lookup** | lookup 是确定性算法（字符串相似度 + 层级过滤），LLM 只参与最终 select；不在 lookup 内调用 LLM |

---

## 11. Path D — 层级实体解析槽填充

Path D 在两个场景中分别验证：**s-011** 用中性 FSM 拓扑验证 adapter 行为；**repair-ticketing** 用业务剧情验证完整工单流程。

### 11.1 s-011 Path D — adapter 行为（抽象场景）

s-011 新增 **Path D**：在 s-011 自己的 FSM 拓扑（`collecting_slots` 状态）中，遇到需要层级实体解析的槽位时，通过 HER lookup + commit 完成槽填充。用中性/抽象的实体层级（不依赖 repair-ticketing 领域数据），专注验证 adapter 与 workingMemory 的读写协议。

**Path D 交互流（抽象层级：`region` → `team` → `member`）：**

```
用户: '帮我找一个人'
  → classify_intent(assign, 0.91) → INTENT_ASSIGN → 进入 collecting_slots

用户: '华东区'
  → FSM 需填 region 槽
  → adapter 读 workingMemory（空），pinned = {}
  → lookup_entity({
      op: 'lookup',
      query: '华东区',
      context: { level: 'region', pinned: {} }
    })
  → candidates: [{ id:'R01', label:'华东区', ... }], suggested: 'R01'
  → LLM 选 selected: 'R01'
  → commit_entity({
      op: 'commit',
      selected: 'R01',
      context: { level: 'region' }
    })
  → adapter 注入 pinned={} → resolved: { id:'R01', label:'华东区' }
  → adapter 写 workingMemory.region = 'R01'

用户: '基础设施组'
  → FSM 需填 team 槽
  → adapter 读 workingMemory → pinned = { region: 'R01' }
  → lookup_entity({
      op: 'lookup',
      query: '基础设施组',
      context: { level: 'team', pinned: { region: 'R01' } }
    })
  → candidates（仅华东区下的 team）, suggested: 'T05'
  → LLM 选 selected: 'T05'
  → commit_entity({
      op: 'commit',
      selected: 'T05',
      context: { level: 'team' }
    })
  → adapter 注入 pinned={ region:'R01' } → resolved: { id:'T05', ... }
  → adapter 写 workingMemory.team = 'T05' → SLOTS_COMPLETE
```

**Path D 验收准则（新增至 s-011）：**
- FSM 状态序列含 `collecting_slots` → `confirming` → `completed`
- `lookup_entity` 每次被调用时，`context.pinned` 包含已 commit 的所有祖先层 id（来自 workingMemory，非 LLM 参数）
- `commit_entity` 成功后，对应 level 的 id 写入 workingMemory
- working_memory 在 `SLOTS_COMPLETE` 时包含所有已填槽位的 id

### 11.2 repair-ticketing Path D — 维修工单完整流程

repair-ticketing example 的 Path D，使用 #162 定义的 FSM 拓扑（`collecting_entities → collecting_description → emit_ticket`）和维修工单业务语言。

**Path D 交互流（维修工单场景，层级：`site` → `department` → `assignee`）：**

```
用户: '我的电脑坏了'
  → classify_intent(repair, 0.89) → INTENT_REPAIR → 进入 collecting_entities

用户: '总部主楼'
  → FSM 需填 site 槽
  → adapter 读 workingMemory（空），pinned = {}
  → lookup_entity({
      op: 'lookup',
      query: '总部主楼',
      context: { level: 'site', pinned: {} }
    })
  → candidates: [{ id:'S01', label:'总部', ... }], suggested: 'S01'
  → LLM 选 selected: 'S01'
  → commit_entity({
      op: 'commit',
      selected: 'S01',
      context: { level: 'site' }
    })
  → resolved: { id:'S01', label:'总部', ... }
  → adapter 写 workingMemory.site = 'S01'

用户: 'IT硬件部门'
  → FSM 需填 department 槽
  → adapter 读 workingMemory → pinned = { site: 'S01' }
  → lookup_entity({
      op: 'lookup',
      query: 'IT硬件部门',
      context: { level: 'department', pinned: { site: 'S01' } }
    })
  → candidates: [{ id:'D02', label:'IT硬件组', ... }, { id:'D03', label:'IT网络部', ... }]
  → LLM 选 selected: 'D02'
  → commit_entity({
      op: 'commit',
      selected: 'D02',
      context: { level: 'department' }
    })
  → adapter 注入 pinned={ site:'S01' } → resolved: { id:'D02', label:'IT硬件组', ... }
  → adapter 写 workingMemory.department = 'D02'
  → 进入 collecting_description

用户: '屏幕无法点亮，已重启两次'
  → description 槽填入 → emit_ticket → 工单生成 → DONE
```

**Path D 验收准则（repair-ticketing）：**
- FSM 状态序列含 `collecting_entities` → `collecting_description` → `emit_ticket` → `completed`
- `lookup_entity` 子层调用时 `context.pinned` 包含已 commit 的祖先层 id（来自 workingMemory）
- `commit_entity` 每次成功后对应 level 写入 workingMemory
- `emit_ticket` 时 workingMemory 包含 `site`、`department` 字段，且值与各 commit resolved.id 一致
- 工单包含 `assignee` 信息（从 department resolved.meta 或后续 assignee 层 commit 获取）

---

## 12. 验收

- [x] 文档落在 `docs/design/162-portable-hierarchical-entity-resolver.md`
- [x] 文档明确 CLI 是 wrapper，进程内调用是 milkie 主路径（第 3、4 节）
- [x] 文档明确 replay/IOPort 不变式风险和规避方式（第 9 节）
- [x] 文档明确祖先拓扑 recall 协议：adapter 从 workingMemory 组装 pinned，commit 后写回（§3.1）
- [x] 文档明确文件路径统一布局及与 #167 的关系（第 1、2 节）
- [ ] `examples/repair-ticketing/` 目录按本设计建立（#162 实现阶段）
- [ ] s-011 Path D E2E 测试通过（#162 实现阶段）
- [ ] repair-ticketing Path D E2E 测试通过（#162 实现阶段）
- [ ] issue #162 body 链接本文档，body 只保留摘要、依赖和验收
