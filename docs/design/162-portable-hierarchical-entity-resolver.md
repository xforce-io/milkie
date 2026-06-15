# #162 Portable Hierarchical Entity Resolver

- **Issue**: #162
- **Branch**: `feat/162-portable-hierarchical-entity-resolver`（或实现分支）
- **状态**: 设计已定稿（#165 评审通过），为 single source of truth；issue #162 body 保留摘要 + 链接指向本文档。
- **落地范围**: `examples/repair-ticketing` 内实现 + s-011 Path D。本阶段不发布独立 npm 包、不承诺稳定公共 SDK、不做跨仓库维护。

---

## 1. 设计定位

**Hierarchical Entity Resolver（HER）不是 AgentRuntime 的核心能力。**

HER 是一个**应用层工具模块**，其职责是：给定用户的自然语言描述，在一个层级化的参考数据集（如维修工单中的「站点 → 楼宇 → 部门 → 负责人」）中，找到最精确的实体节点并校验选择合法性。

它的正确归属：
- 放在 `examples/repair-ticketing/resolver/` 下，作为该 example 的领域模块。
- 对 milkie 的 `AgentRuntime`、`FSMEngine`、`IOPort` **零依赖**。
- milkie adapter（第 3 节）是它在 FSM 工具 handler 中的薄接线层，不是它的一部分。

**为什么不内置进 AgentRuntime？**

AgentRuntime 管理 FSM 状态、LLM 调用和 IOPort 边界。把领域专属的层级数据遍历逻辑内置进来会模糊 runtime 的职责边界，且 HER 的数据加载（CSV/JSON）与 replay 不相关，强行接线反而引入不必要的不确定性记录负担。

---

## 2. 架构边界

```
┌─── examples/repair-ticketing ─────────────────────────────────┐
│                                                                 │
│   milkie agent (FSM + slot filling)                            │
│       │                                                         │
│       └──► milkie adapter                                       │
│            (tool handler, src/tools/entity-resolver.ts)        │
│                 │  in-process call (no subprocess)              │
│                 ▼                                               │
│           EntityResolver (pure TS module)                       │
│           resolver/EntityResolver.ts                            │
│                 │                                               │
│                 ├── resolver/schema.json   (hierarchy schema)   │
│                 └── resolver/data.csv      (entity data)        │
└─────────────────────────────────────────────────────────────────┘

          ┌─────────────────────────────────────┐
          │  CLI wrapper (scripts/resolver.ts)  │
          │  (thin shell over same EntityResolver)│
          │  stdin → JSON → EntityResolver      │
          │          → JSON → stdout            │
          └─────────────────────────────────────┘
          ↑
          供 AgentScope / Python / 非 TS 框架调用（第 4 节）
```

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
    return resolver.lookup(input)
  },
}

export const commitEntityTool = {
  name: 'commit_entity',
  async handler(input: CommitInput, ctx: ToolContext) {
    return resolver.commit(input)
  },
}
```

`EntityResolver` 本身是**纯同步数据处理**（索引在构造时建完），handler 的 async 包装仅满足 milkie 工具接口约定。

---

## 4. CLI Wrapper — 非 TS 框架的薄门面

CLI wrapper 是同一个 `EntityResolver` core 的**命令行入口**，不包含任何额外逻辑。

```
node scripts/resolver.ts <command>
```

通信协议：**单次 stdin → stdout JSON**（无长连接、无状态），调用方负责进程管理。

```
stdin:  { "op": "lookup", ...payload }
stdout: { ...result }

stdin:  { "op": "commit", ...payload }
stdout: { ...result }
```

**CLI 不是 milkie 的主路径。** milkie agent 进程内已有 `EntityResolver` 实例，不会通过 spawn 子进程调用 CLI。CLI 仅供以下场景：
- AgentScope / Python-based 框架集成
- 快速命令行调试：`echo '{"op":"lookup","query":"三楼机房"}' | node scripts/resolver.ts`

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
    pinned?: Record<string, string>  // 已固定的上级节点 id，如 { site: 'S01', dept: 'D03' }
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
    pinned?: Record<string, string>
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
1. FSM action handler 调用 lookup(query, context)
         ↓
2. lookup 返回 { candidates, options, suggested }
         ↓
3. LLM 收到上述结果作为工具返回值，在 collecting_slots 状态下生成下一步工具调用
         ↓
4. LLM 调用 commit_entity({ selected: "<从 options 中选出的 id>", context })
         ↓
5. commit 做硬校验（selected ∈ options），返回 resolved（+ 可能的 corrected）
```

**LLM 是唯一负责"选择"的角色。** HER 提供候选集合和建议，但不替 LLM 做最终决策（`suggested` 只是参考）。`commit` 的硬校验确保 LLM 不能幻觉出一个不存在的实体 id。

---

## 7. `corrected` 语义

`corrected` 是 `commit` 在以下情况下返回的字段：

> **`selected` 是合法的 id，但与 `context.pinned` 中的约束冲突。**

示例：

- `context.pinned = { dept: 'D03' }` 表示该工单已由业务规则固定派给 D03 部门。
- LLM 选择了 `selected: 'D07'`（一个合法 id，但不是 D03）。
- `commit` 发现冲突 → 返回 `{ resolved: <D03的实体>, corrected: 'D03' }`。

语义约定：
- `corrected` **存在** → 调用方应用 `corrected` 覆盖 `selected`，向用户解释"该字段已由系统固定"。
- `corrected` **不存在** → `selected` 被直接采纳，`resolved` 是最终实体。
- `corrected` 不用于纠正 LLM 的"猜测偏差"，只用于执行**业务侧的固定约束**（pinned）。

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
| `levels[].name` | 层级逻辑名（协议中 `context.level` 的合法值） |
| `levels[].idColumn` | data.csv 中作为 id 的列名 |
| `levels[].labelColumn` | data.csv 中作为显示名称的列名 |
| `levels[].parentLevel` | 父层级名（用于过滤：指定 dept 后只展示该 dept 下的 assignee） |
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
| **DAG / 多级并发解析** | 本阶段只做单条路径的线性层级解析（site → building → dept → assignee），不做多路分支 DAG |
| **同一工单多值 / 同级多值** | 每个层级只选一个节点；不支持一张工单同时分派给两个部门 |
| **训练式 NER** | lookup 走确定性字符串匹配 + 评分，不引入模型推理，不做实体标注训练数据生成 |
| **tool-bearing skill** | HER 不以 milkie skill（.md 技能文件）形式暴露工具；工具注册在 example 的 agent setup 代码里 |
| **独立 npm 包发布** | 本阶段边界：example 内实现，不建立独立 package + 版本管理 |
| **跨仓库维护承诺** | 代码住在 monastery repo 的 example 目录，未来抽取复用以本文档边界为参考 |
| **LLM 驱动的 lookup** | lookup 是确定性算法（字符串相似度 + 层级过滤），LLM 只参与最终 select；不在 lookup 内调用 LLM |

---

## 11. s-011 Path D — 层级实体解析槽填充

s-011 新增 **Path D**：FSM 在 `collecting_slots` 状态中，遇到需要层级实体解析的槽位（如"分派部门"），通过 HER 的 lookup + commit 完成槽填充。

**Path D 交互流（维修工单场景）：**

```
'我的电脑坏了'            → classify_intent(repair, 0.89) → INTENT_REPAIR
                          → 进入 collecting_slots

'是哪个部门负责维修？'    → FSM 需填 assignee 槽
                          → lookup_entity({ query: "电脑维修", level: "department" })
                          → candidates: [IT硬件组, IT软件组], suggested: "IT硬件组"
                          → LLM 选 selected: "D02"
                          → commit_entity({ selected: "D02" })
                          → resolved: { id: "D02", label: "IT硬件组", ... }
                          → 槽 assignee 填入 D02 → SLOTS_COMPLETE

'确认'                    → confirm_action(true) → USER_CONFIRMED
                          → spawn repair-executor → DONE → completed
```

**Path D 验收准则（新增至 s-011）：**
- FSM 状态序列含 `collecting_slots` → `confirming` → `executing` → `completed`
- `lookup_entity` 被调用 1 次，返回非空 `candidates`
- `commit_entity` 被调用 1 次，`resolved.id` 存在且在 `candidates` 中
- working_memory 在 `SLOTS_COMPLETE` 时含 `assignee` 字段
- `repair-executor` sub-agent 被 spawn 且 `TaskResult.success`

---

## 12. 验收

- [x] 文档落在 `docs/design/162-portable-hierarchical-entity-resolver.md`
- [x] 文档明确 CLI 是 wrapper，进程内调用是 milkie 主路径（第 3、4 节）
- [x] 文档明确 replay/IOPort 不变式风险和规避方式（第 9 节）
- [ ] `examples/repair-ticketing/` 目录按本设计建立（#162 实现阶段）
- [ ] s-011 Path D E2E 测试通过（#162 实现阶段）
- [ ] issue #162 body 链接本文档，body 只保留摘要、依赖和验收
