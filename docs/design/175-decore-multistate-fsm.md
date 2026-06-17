# 去 DialogFlow / 多态 FSM：core runtime 轻量化

- **Issue**: [#175](https://github.com/xforce-io/milkie/issues/175) — 去 DialogFlow / 多态 FSM：core runtime 轻量化（双层状态机 + 四信号接口）
- **状态**: 评审中（设计 + spike 已完成，待批准后进入实现）
- **分支**: `feat/175-decore-multistate-fsm`
- **关联**: 受影响 story [s-011](../stories/s-011-multi-state-fsm-intent-routing-and-slot-filling.md)；作废 roadmap 项 #60 / #31；迁移靶子 `examples/repair-ticketing/`；设计背景参照 `ARCHITECTURE.md`（run-as-product 不变量）与 TIANSU 架构 §6.8（slot-filling 正交化的外部佐证）

本文档是该变更的 single source of truth。issue body 仅保留摘要 + 指向本文档的链接；issue 评论保留评审讨论与决策过程。

---

## 1. 目标与范围

把 core 从「**多态 FSM runtime**」收敛为「**单循环自主 runtime + 最小运行生命周期状态机**」，core 对**业务阶段保持无知**；删除开发者 authored 的多态业务 FSM 及其确定性债，同时**不牺牲主力场景**。

**问题陈述**：当前 core 的 `FSMEngine` 提供「开发者 authored 的多态 FSM」（intent routing / DialogFlow / multi-state workflow，见 s-011）。这台「业务拓扑 FSM」：

1. **对确定性是净债务而非资产**。milkie 的 run-as-product（回放/fork/diff/lineage）建立在 **IOPort + event log** 上，与控制流状态机正交——`ReplayingIOPort` 回放的是 effects，不依赖 `FSMEngine`。而 tool handler 里的 `ctx.emit` 驱动转移是**非确定性副作用**，反而逼出 **#60**（`tool.emitted` + `replayEmits`）与 **#31**（guard 回放）这些「为了让自己能回放」的补丁。
2. **唯一站得住的用途是受监管 dialog-SOP**（强制顺序 / 必访阶段 / 模型外强制 + 可审计），属治理/合规域（需 Security + HITL + 审计），是 milkie 明确的**非目标**；milkie 缺这两根支柱，本就无法完整服务该场景。
3. **其余场景**（自主 / slot-filling / sub-agent / intent routing / 非监管结构化对话）都**不需要**多态 FSM。

**In（core 保留 / 重构）**
- 下层 **运行生命周期 SM**（`RunLifecycle`）：`running / paused / interrupted / completed / failed`，框架固定、与业务无关。
- **四信号接口** `RunResult`：`continue / need_input / done / error`，loop body → 生命周期 SM 的唯一上行通道。
- 两个对业务无知的 hook：可 checkpoint 的 userland 状态（WM 已具备）、每轮 `assemble` + outcome hook。

**Out（删除 / 降级 / 不做）**
- 删除 `FSMEngine` 多态部分（states map / `on:` / `pendingEvent` / emit→transition）。
- 删除 **#60**（`tool.emitted` / `replayEmits`）与 **#31**（guard）——随多态 FSM 一并作废；runtime **写路径**不保留 deprecated 模式，历史读路径兼容见 D6/D7。
- 上层「业务拓扑 SM（DialogFlow）」**降为 userland 组合模式**；core 不提供、不禁止。受监管 SOP 出 scope。
- 非监管结构化对话走更轻的分档（见 §6），不进 core。

---

## 2. 决策摘要

| # | 决策 | 理由 |
|---|---|---|
| **D1** | 上层业务 SM（DialogFlow）**只留 userland + example**，core 不留 `PhaseDriver` helper | core 对业务阶段无知；受监管 SOP 出 scope 且 milkie 无 Security/HITL 服务不全 |
| **D2** | 新 runtime **停止写** `tool.emitted` / `fsm.transition` 事件类型并删除 #60/#31 | 它们是多态 FSM 的确定性债，随之作废；历史读路径兼容由 D6/D7 约束 |
| **D3** | 断面阈值**实测优先**，不预设 N | spike 已证断面收敛、可解释（见 §7） |
| **D4** | diagnosable 走 **(a) 在 llm/tool effects 上重建**，不下放 userland、不缩水 | diagnose 是 milkie 核心能力（反馈验证支柱）；重锚后覆盖所有 agent，比今天更通用 |
| **D5**（spike 新增） | de-core 必须**拆分 `transition()`**：reserved 重入保留进 `RunLifecycle`，只删用户态→用户态业务转移 | spike 实测：resume/recovery 与业务多态**共用** `transition()`，盲删会打断 resume（见 §5） |
| **D6**（评审新增） | 删除旧 FSM 事件前，先落地新 diagnosable 锚点并短期双读/双写验证 | 避免 `fsm.transition` 消失后 trace/viewer/explain 出现诊断空窗 |
| **D7**（评审新增） | runtime 不再写旧 checkpoint schema，但必须继续读旧 `agent.checkpoint.payload.checkpoint.fsm` | run-as-product 要保住既有 event log / portable session / serve restart recovery |

---

## 3. 双层状态机模型

去 core 化后，系统不是「去掉状态机」，而是**解耦两台被错误混在一起的状态机**——删上层，留下层。

| | 下层 · `RunLifecycle`（**留，core 必选**） | 上层 · 业务拓扑 SM（**删出 core**） |
|---|---|---|
| 状态来源 | 框架固定 | 开发者 authored |
| 状态集 | `running / paused / interrupted / completed / failed` | `intake / quote / …`（任意） |
| 建模 | 一个 run 的执行生命周期 | 任务/对话结构 |
| 转移触发 | 框架信号 | tool 的 `ctx.emit` |
| 对确定性 | **资产**（interrupt/resume/checkpoint/replay 依赖它良定义） | **债**（#60/#31 为它擦屁股） |

> 今天二者被错误地混在同一个 `FSMEngine`（`paused/error_handling/failed` 是 reserved states，`fsm.emitEvent('interrupt')` 走同一引擎）。本变更 = 解耦。

**两层关系（仅 DialogFlow 这一支需要时）**：业务态与 lifecycle **正交、非细化**——业务态在 `paused` 期间仍有值且被保留（resume 回到原业务态而非重头）。依赖**单向**：上层硬依赖下层的 pause/resume/checkpoint（机制依赖）；下层只依赖信号协议，**永不引用业务态名**。状态定义**零泄漏**是 de-core 后要守住的不变量（泄漏 = 退回今天的 reserved-state smell）。

- 上层 **run-scoped**（workflow 型，一个 run 内跑完）→ 是 `running` 的细化（无 history）。
- 上层 **session-scoped**（DialogFlow 型，跨 turn）→ 与 lifecycle 正交、被 `running` 活动门控、跨 lifecycle 态保留（需 deep-history）。

---

## 4. 四信号接口 `RunResult`

上层（含 slot-filling、DialogFlow、自主循环）与下层只通过一条窄缝耦合：

```
上层 / loop body ── RunResult ──▶ 下层 RunLifecycle
   continue   → running → running
   need_input → running → paused（写 checkpoint {lifecycle, userland blob}）
   done       → running → completed
   error      → running → failed

下层 ──▶ 上层：仅 activation / restoration（resume 时先 restore 再激活；从不查看业务态）
```

自主 agent = 上层退化成单态，但**吐的还是这四信号**——这正是下层能当稳定 core、上层（含 slot-filling/DialogFlow）都是同一协议生产者的原因。**接口不变，身后实现可换**。

---

## 5. `transition()` 拆分（D5 · spike 头号发现）

**实测发现**：resume / error-recovery 重入用户态，走的是与业务多态**同一个** `transition()`：

```
cross-user-state transition "paused" -> "react" via "RESUME"
```

即**连单态自主 agent 的「中断→恢复」也复用 FSM 转移机制**。因此不能盲删整个 `transition()`，必须按**「从哪来」**拆分：

| 转移模式 | 性质 | 处置 |
|---|---|---|
| reserved → 用户态（`paused→X` via RESUME、`error_handling→X`） | 生命周期重入/恢复 | **保留**进 `RunLifecycle`（restore + 重激活） |
| 用户态 → 用户态（`classify→handle_a`、`start→end`） | 业务多态 | **删除** |
| 用户态 → reserved（`X→paused` via interrupt、`X→error_handling`） | 生命周期挂起/降级 | **保留**进 `RunLifecycle` |
| 自循环（`X→X`） | 单态推进 | **保留**（= 循环继续，`continue` 信号） |

`RunLifecycle` 接管前三类（全部只涉及 reserved 态与「当前用户态」，不需要业务拓扑）；被删的只有「用户态→另一个用户态」这一类。

---

## 6. 结构化对话分档（D1 配套）

非监管结构化对话**不需要状态机**，按机制递增、能停就停（TIANSU §6.8 对此独立佐证：slot-filling 是正交 wrapper、规则 vs 事实分家、软引导 + 硬地板）：

| 需求 | 最小机制 | 是状态机吗 |
|---|---|---|
| 阶段聚焦 / 软顺序 | prompt（软引导） | 否 |
| 槽位完整性（类型/必填/格式/枚举） | tool param schema（硬地板，本地「规则」校验） | 否 |
| 跨 turn 阶段记忆 / 每阶段 loadout | WM 里一个 phase 变量 + 条件 `assemble` | 否 |
| 一两条硬门（Y 成立才能 X） | action precondition | 否 |
| 硬拓扑 / 必访 / 可审计序列 | 上层 SM | **是，且仅监管 SOP（出 scope）** |

> milkie 版「Param Guard 等价物」只做**本地规则校验**（schema）+ 自包含字典候选解析；**本体存在性校验**与 **Security 行级权限过滤**因缺对应支柱而**出 scope**（凡权限敏感候选集纯 milkie 服务不了，需诚实标注）。slot-filling 作为四信号生产者挂到下层：某槽 missing/ambiguous → `need_input` → `paused`。

---

## 7. 验证：spike 实测断面

在 `FSMEngine.transition()` 注入「业务多态即抛」探针（放行 reserved 重入与自循环），在**离线集**（709 unit，含 cluster A/B 单测）实测。探针完成后已**完全撤回**，baseline 复绿（53 unit + 8 deterministic e2e）。

> 全量 `test:all` 不可用——被大量 live e2e 超时（缺 key 挂住）污染。离线集才是干净基准。

**真实业务多态断面 = 9 suites / 27 tests**，实际触发的转移仅 `classify→handle_a`（intent routing）与 `start→end`（action 链）两类：

| 簇 | suite | 处置 |
|---|---|---|
| A 引擎/运行时 | `FSMEngine` · `AgentRuntime` · `AgentRuntime.currentTurn` · `Replay`(#60) | 删 / 重构 |
| B diagnosable（live-trace 子类） | `CausedByGraph` · `Trace` | 重锚 effects |
| 意外 | `standardAgentLayer`（`start→end` action 链） | **多态不止 DialogFlow**，简单 action 链也用 |
| 迁移靶子 | `examples/repair-ticketing` ×2 | 改 slot-filling + precondition |

**两个「松一口气」**（初始担心是错的）：排除 reserved-from 误判后，`checkpoint/resume`（`checkpoint-from-events`）与 `serve` **全部存活**——证明它们只依赖生命周期态，与业务多态无关。

**diagnosable 重建分两条战线**（细化 D4）：
1. **live-trace 生产者**（`CausedByGraph`/`Trace`）——随运行时改动断，重锚到 llm/tool effects（`explainTransition`→`explainDecision`，锚 `llm.responded`）。
2. **fixture-trace 读者**（`render-tree/html/viewer` · `explainTransition` · `fsmStateAt`）——运行时探针下**不断**，只在**删除 `fsm.transition` 事件类型**时才断，且其**录制 fixture 需迁移**。`fsmStateAt` 为 FSM 专属，大概率删除。

**删除旧事件前的硬门槛（D6）**：
- 先定义并落地新诊断锚点（暂名 `decision.point`，最终命名以实现为准），语义为「一次 loop body/outcome 决策」，锚到已落日志的 `llm.responded` / `tool.responded` effects，而不是业务态转移。
- 在删除 `fsm.transition` 写入前，`explainDecision` / HTML / viewer 必须能只依赖新锚点解释：触发输入、相关 tool result、outcome（continue / need_input / done / error）、因果链。
- 迁移期允许**短期双写**（旧 `fsm.transition` + 新 decision 锚点）或**双读**（新读者优先读 decision，缺失时读旧 `fsm.transition`）。双写只用于验证，不进入长期 public contract。
- 删除旧写入的同一个 PR/切片内，必须保留旧 trace **读路径兼容**：历史 `fsm.transition` 仍可在 HTML/viewer 中显示，`explainTransition` 可降级为 legacy adapter 或被 `explainDecision` 包装读取。
- 验收用例：同一条 intent-routing 录制，在旧锚点与新锚点上得到等价的 causedBy chain、触发工具、最终 outcome；删除旧写入后 live trace 不丢 Why?/decision 展示。

---

## 8. Checkpoint / Event-log 迁移契约（D7）

runtime 可停止**写**旧 `fsm` checkpoint schema，但不能停止**读**旧 event log。`agent.checkpoint` 是 resume 的事实来源，已有 runs / portable session / serve restart recovery 都可能包含：

```ts
checkpoint: {
  fsm: {
    currentState: 'paused',
    resumeState: 'react',
    stateData: null
  },
  context: { workingMemory, regions },
  ...
}
```

新 schema 采用并列结构（字段名可在实现中微调，但必须显式 version）：

```ts
checkpoint: {
  schemaVersion: 2,
  lifecycle: {
    status: 'paused',
    resumeKind: 'loop',
  },
  userland: {
    // opaque, checkpointable userland blob; absent for default autonomous loop
  },
  context: { workingMemory, regions },
  ...
}
```

**兼容规则**：
- `checkpointFromEvents` / `loadCheckpoint` 必须支持 v1 和 v2：有 `schemaVersion >= 2` 读 `lifecycle`；无 version 且有 `fsm` 视为 v1 legacy。
- v1 `fsm.currentState === 'paused' && fsm.resumeState` 映射为 `lifecycle.status='paused'` + `resumeKind='legacy-state'`，恢复时进入默认 loop activation；不把 `resumeState` 暴露为业务态给新 runtime。
- v1 非 paused checkpoint（历史 continuation checkpoint）映射为 `lifecycle.status='running'` 或 `completed` 的实现侧等价状态，原则是保持可继续/可检查，不要求复活旧业务拓扑。
- 新 runtime 不再写 `fsm` 字段；如需兼容外部读取，可只在投影/CLI 层把 `lifecycle` 格式化成 legacy-looking summary，不能把旧 schema 写回 event log。
- 删除 `FSMEngine.snapshot/restore` 前，先加迁移单测：旧 interrupt checkpoint 可 resume、旧 continuation checkpoint 可被 `checkpointFromEvents` 读出、serve restart recovery 仍通过。

---

## 9. 影响、非目标与风险

**影响分级**
- **零损失**（换表达、不丢能力）：run-as-product 全家桶、自主 / slot-filling / sub-agent / interrupt-resume / checkpoint。
- **搬到 userland**：多态 authored 功能（s-011 / DialogFlow / multi-state workflow / action 链），authoring 接口变。
- **唯一「牺牲」**：受监管 dialog-SOP——本就出 scope 且 milkie 无 Security/HITL 服务不全。

**净收益**：随多态 FSM 一并甩掉 #60/#31 的确定性债；userland 上层 SM 因决策在 loop body（读已落日志的 `llm.responded`）而**天然 replay-clean**，不再制造新债。

**非目标**：治理 / 语义建模 / 受监管对话 SOP（天枢 scope）；core 不内建 `PhaseDriver`。

**风险与缓解**
- *diagnosable 重锚回归*：先固定一组「explain a decision」黄金用例，重锚前后断言一致。
- *fixture 迁移遗漏*：删 `fsm.transition` 事件类型时，全量 grep fixture jsonl + 投影读者，逐个迁移。
- *历史 checkpoint 不可恢复*：按 §8 保留 v1 读路径，新增旧 event-log fixture 回归。
- *action 链迁移*：`start→end` 这类确定性链改为单 action 态内顺序执行，或 userland `PhaseDriver`。

---

## 10. 实现切片顺序（按 spike + review 校准）

1. **先落新 diagnosable 锚点**（D6）：新增 decision/outcome 投影与黄金用例；HTML/viewer/explain 先双读，live 路径可短期双写验证。此时旧 `fsm.transition` 仍存在。
2. **抽 `RunLifecycle` + checkpoint v2 读写**：按 §5 拆 `transition()`——reserved 重入/挂起 + 自循环保留进 `RunLifecycle`；新增 `{schemaVersion, lifecycle, userland}` 写入，同时保留 v1 `fsm` 读路径（§8）。
3. **删业务多态 FSM + #60/#31**：删除用户态→用户态业务转移、`pendingEvent`、`tool.emitted`、`replayEmits`、guard；迁移 A 簇测试。此切片不得删除旧 trace 读路径。
4. **删除新 runtime 的旧事件写入**：停止写 `fsm.transition`；保留 legacy renderer/adapter 读取历史 trace；迁移 fixture-trace 读者，删除或降级 `fsmStateAt`。
5. **迁移 `examples/repair-ticketing`**：DialogFlow → slot-filling + action precondition，作为「轻量分档」POC。
6. **两个 hook + userland `PhaseDriver` example**：每轮 `assemble`/outcome hook、可 checkpoint userland 状态；出参考实现 + 一个受监管子流程示例（如适当性评估）证明「结构化/监管对话不进 core 也能搭」。

每个切片走 TDD + 绿色基线，断面对照本文档 §7。
