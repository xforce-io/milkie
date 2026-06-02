# Goal: Lineage 驱动的引用溯源（替代前端正则启发式）

> **一句话目标:** 让 agent-docs-qa 的 citation / provenance **完全由 trace 中的
> `object.created` / `relation.created` 事件驱动**，彻底删除前端"读心"agent
> 文本的正则启发式（`CITATION_RE` / `classifySegment` / substring 匹配）。
>
> **任务链:** #39 → #37 → #38 → 【新】agent 结构化 cite 产出 → #40
> （均 Parent: #20）。本文是这条链的北极星，每个 issue 实现时对照此处验收。

---

## 1. 为什么做（问题陈述）

当前 `examples/agent-docs-qa/public/index.html` 用前端正则反推引用：

```
grep/read_file → agent 看到内容
              ↓
agent 生成自由散文，凭记忆写「引文」（引用：file:line）
              ↑ lossy re-encoding：转简体 / 记错行号 / 合并诗句 / 改标点
              ↓
前端 CITATION_RE 解析文本 + 把 quote 拿去 corpus 做 substring 匹配
              ↑ provenance 误报全部在这里诞生
```

**根本缺陷:** provenance 是从 agent 的**文本复述**反推的，而文本复述是 agent
对所读内容的有损重编码。前端在"读心"——事后猜 agent 引用了什么，永远追不上
文本的各种变形。

我们已经在 #108 用 opencc 繁简归一化 + 全文回退（misattributed）+ 去空白匹配
堵了三类误报，但那是**打地鼠**：每堵一种变形，下一种（标点）还会冒出来。而且
这些代码本身就是 #40 标记的**待删债务**——本目标达成时，连同它们一并 burn-down。

## 2. 概念模型：event → object → relation

理解整条链先要分清这三层（**全部都是 trace 里的 event**，区别只是职责）：

```
event        append-only 流水账的原子记录（llm.requested / tool.responded / …）
  └─ object.created    也是一种 event：在某条 event 产出的内容上「圈框 + 铸 objectId」
        producerEventId ──> 指回内容来源（如 read 的 tool.responded）
        meta            ──> {file, lineStart, lineEnd}（结构化出处，非 agent 文本）
        └─ relation.created   又是一种 event：在两个 objectId 之间连一条有类型的边
              from/to ──> 两端都是 objectId（cites: claim → passage）
              causedByEventId ──> 挂回因果链
```

- **event** 是地基（一切都是它，append-only、replay-canonical）。
- **object** 是 event 之上的**句柄层**：把流水账里一段原始内容，提升成有 `id`（不可
  伪造句柄）、有 `type`（#39 taxonomy）、有定位 `meta` 的可引用实体。
- **relation** 在 object 之间连边，合成一张 **lineage 图**；provenance 查询 = 在图上
  沿边走（claim ──cites──> passage ──meta──> file:line），**不在文本上猜**。

> 这套 event/object/relation 概念模型 **ARCHITECTURE.md §Concept Model 现在还没有**
> （只有 `Event`/`Trace` 等子节，无 `Object`/`Relation`）。补上它正是 **#39** 的职责
> （见 §6）。本节是 goal 内部的工作定义，最终以 #39 落定的 taxonomy / ARCH 子节为准。

## 3. 设计原则

1. **Producer 显式声明，runtime 记录事件；绝不解析 LLM 文本反推**
   （#37/#38 的 non-goal 红线）。

2. **引用的不可靠性分两层，分别处理:**

   | 层 | 含义 | 本目标如何处理 |
   |---|---|---|
   | 表征层 | agent 怎么*指代*来源（写 `file:26` + 复述引文） | 用 runtime 铸造的 **objectId 句柄**取代 → 造假空间归零 |
   | 认知层 | agent *判断*"这句是否真基于该来源" | 提供 claim↔object 结构化锚点 → 错误变得**可审计**，交给 verifier/diagnoser 核验（本线不做核验） |

3. **objectId 是不可伪造的句柄（capability-like）:** `file:26` 是 agent 能凭空
   拼的字符串；`objectId` 是 runtime 在 agent **真读过之后**才发给它的 token。
   agent 只能"出示"持有的句柄，编不出假出处。这是表征层造假归零的根。

## 4. 北极星验收（End-to-End Definition of Done）

整条链完成时，下列**全部**成立:

- [ ] `index.html` 中**不存在** `CITATION_RE` / `classifySegment` /
      `hasVerbatimSubstring` / `extractQuotedSnippets`，以及 #108 引入的
      `normForMatch` / opencc 归一化 / misattributed / 去空白匹配。
- [ ] citation chip、Sources footer、Provenance tab **全部由
      `object.created` / `relation.created` 事件投影渲染**，前端零 substring 匹配。
- [ ] agent prompt **不再要求** agent 在文本里写 `chapter:行号`；引用是
      结构化声明（见 §6 新 issue）。
- [ ] Provenance 状态从"匹配置信度代理"（supported/paraphrased/misattributed/
      tenuous）变为**事实性**两态:有 `cites` 关系且指向真实 read object =
      **grounded**；无 = **model-generated**。
- [ ] `Milkie.replay(runId)` 的 cache 序列**不变**，lineage 事件**不引入**
      任何 live I/O（#37/#38 的硬约束）。

**回归验收（用真实踩过的 case 证明根治）:** 下列三个历史误报场景，在 lineage
模型下**概念上不可能发生**，因为不再有"匹配"这个动作:

| 历史 case | 旧根因 | lineage 下 |
|---|---|---|
| ch-49 行号标错 `:26` | agent 文本里手写错行号 | 出处 = read 工具的真实 range，非笔误 |
| ch-01 繁简失配 | agent 简体复述 vs 繁体语料 | 引文显示 object 的真实 bytes，不比对文本 |
| ch-50 跨行诗合并 | agent 合并诗句致 substring 失配 | object 是读取的内容块，无匹配动作 |

## 5. 里程碑（M1 / M2，增量交付）

分两步落地，不必一口气全做。M1 不碰 agent、不赌 LLM 配合，先验证"object 事件能
驱动 UI"；M2 才是完整目标，风险集中在【新 issue】。

| 里程碑 | 含 issue | 交付 | 不含 |
|---|---|---|---|
| **M1 — run 级** | #39 + #37 + #40(footer 部分) | Sources footer「Sourced from chapter-01.txt」由 read 的 `object.created` 事件驱动，**零 agent 改造**，删掉一部分前端正则 | 逐段 provenance |
| **M2 — 段落级** | #38 + 【新】+ #40(provenance 部分) | 逐 claim 的 `grounded` / `model-generated`，由 `relation.created(cites)` 把 claim 绑到 passage object，**删尽**前端正则 | —— |

**为什么这么切：** M1 的价值是用最小代价(无 agent 改造、不依赖 LLM 配合)证明
"事件驱动 UI"这条路通，并 burn 掉一部分 #40 债务；M2 才需要 agent 结构化产出
cites（§6 新 issue），那是整条链风险最高的一环。

## 6. 任务分解与各自验收

> 顺序即依赖序：#38（relation 能力）必须在【新 issue】（用该能力）之前。

### #39 — artifact 类型 taxonomy 定稿（设计 doc，先做，无依赖）
**DoD:**
- `docs/lineage-taxonomy.md` 评审通过；object type 含 `passage`、relation type 含
  `cites`，字段含 `file/lineStart/lineEnd/hash`。
- **在 ARCHITECTURE.md §Concept Model 新增 `Object` / `Relation` 两个子节定义**
  （对齐现有 `Event`/`Trace` 子节"定义 + *Not:*"写法）——现在 §Concept Model 只有
  Event/Trace 等、**无 Object/Relation**，本 issue 负责补上（见 §2）。不是加个链接，
  是补一等定义。
- 指出三国 example 的 emit hook 位置（如 corpus `read_file` 处，不要求实现）。

### #37 — `object.created` + tool 关联（依赖 #39；#25 已 done）
**DoD:**
- 新增 `EventKind: 'object.created'`，payload 含 `objectId / type / producerEventId
  / hash? / meta`。
- `ToolContext`（`src/types/tool.ts:7`，**已经传进 handler**）加 `createObject(...)`
  方法——往现有接口加方法，**不是新建注入**；`buildToolContext`
  （`AgentRuntime.ts:358`）是实现处。`producerEventId` 指向真实 `tool.responded`，
  `hash` 与 #25 的 `outputHash` 对齐。
- **corpus 工具实质改造（易漏，务必写明）:** `read_file` 现在读整文件、不收行号、
  不返 id（`corpus-tools.ts:33`）；需改为**支持行号范围** + **tool result 回传
  objectId**，`grep` 同理。否则 agent 没有可 cite 的精准 passage 句柄。
- replay cache 序列不变。

### #38 — `relation.created` 事件（依赖 #37/#39；**排在【新】之前**）
**DoD:** 新增 `EventKind: 'relation.created'`，payload 含 `relationId / type /
fromObjectId / toObjectId / causedByEventId`；`ToolContext` 加 `createRelation(...)`
producer API；`from/to` 均可解析到 `object.created`，`causedByEventId` 可解析到生产
事件；replay 序列不变。本 issue 是**能力**，只需一个最小 producer 满足 acceptance；
真正的 producer（sanguo-researcher）由下面【新 issue】承载。

### 【新 issue】sanguo-researcher 结构化引用产出（依赖 #37/#38；**当前 gap，风险最高**）
> 现有 issue 未认领此环:#38 的 acceptance 只要"至少一个 producer 能 emit"
> （toy 即满足），#40 的 scope 只是前端消费——**没人负责让 agent 真的改造**。
> 这是整条链成败的关键，必须独立成 issue。

**DoD:**
- sanguo-researcher 从"单 llm state + 自由文本写 `file:line`"改为**结构化声明
  cites**:用 #37 回传的 `objectId`，通过 `cite(objectId)` 工具显式产出
  `relation.created(cites)`。
- **选型（代码现实决定）:** 走**路线 A — `cite(objectId)` 工具**，复用现有
  `ctx.createRelation` + 工具调用机制，无需新基建。弱点是 agent 可能漏调，靠 prompt
  强约束（必要时加"answer 前必须 cite"的 FSM 守卫）。**不走 answer schema**——`src/`
  当前无 structured output 机制（`responseFormat`/`outputSchema`），schema 路线要从零
  建基建，列为未来选项。
- 确立并测试约束:**objectId 由 runtime 铸造、不可伪造**；agent 只能引用收到过的
  id，无法凭空声明出处（不可伪造性测试）。权衡见 §3。

### #40 — agent-docs-qa 前端正则清算（依赖以上全部，**最终验收**）
**DoD:** §4 北极星验收的全部勾选项；Sources/Provenance tab 行为不变（但来自
lineage 事件而非启发式）；删尽 §4 列出的所有正则/匹配代码。

## 7. 不变量与约束

- **不解析文本反推**（贯穿 #37/#38/新 issue/#40）。
- **objectId 不可伪造**:runtime 铸造，agent 仅持有/出示。
- **replay determinism 不变**:lineage 事件是记录，不新增 live I/O。
- **CLI/UI 是投影**:provenance 渲染消费事件投影，不是平行事实源（ARCH invariants）。

## 8. 非目标（明确划界）

- **不做语义核验**:"这段 claim 是否*真的*被 object 支撑"是 LLM 认知层判断，
  本线只提供锚点，核验交给 verifier / diagnoser（#35/#36）。
- **不引入 DB / 独立 lineage service**（#37/#38 non-goal）。
- **不在本线做 reverse index / lineage query**（#41）。
- **不追求消除认知层错误**:agent 仍可能 cite 错 object；区别是错误从"不可审计
  的字面失配"变为"可核验的语义错误"。

## 9. 关联

- Parent: #20（Trace substrate gap）
- 依赖链: #39 → #37 → #38 →【新】→ #40
- 前置已完成: #25（tool.responded metadata / outputHash）
- 被本线 burn-down: #108 的前端 provenance 启发式（opencc 归一化 / misattributed /
  去空白）
- 下游: #41（reverse index）、#42（lineage UI）、verifier/diagnoser（#35/#36）
