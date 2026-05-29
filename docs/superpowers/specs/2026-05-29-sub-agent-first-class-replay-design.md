# sub-agent 一类公民化：独立 runId + 独立可 replay（#47）

**Issue:** #47 [replay/fork P0] sub-agent 一类公民化：独立 runId + nested sub-trace + 递归 replay
**Parent:** #20（Trace substrate gap）
**Depends on:** #24（父锚点 agent.spawned/returned + childRunId 字段，已合并）
**Date:** 2026-05-29

## 背景

#24 落地了 ARCH invariant #11 目标态的前半「one event」——父 run emit `agent.spawned` /
`agent.returned` 锚点。本 issue 落地后半「nested sub-trace」：让子 agent 成为一类公民
（独立 runId、自己的 `<childRunId>.jsonl`、自己的 `agent.run.started/completed`、可被独立
replay）。

### 现状（被 #47 修复的洞）

子 agent 经 `makeSubAgentTool` 调用，调用本身走父的 `ioPort.invokeTool`
（`AgentRuntime.ts:1066`）。spawn 时子复用父的 `agentRunId` 与父的 `RecordingIOPort`
（`AgentRuntime.ts:242,245`）：

- 子的 llm/tool/clock/uuid 事件全 append 到**父的** runId，挤在父的一条 jsonl，没有独立
  sub-trace；
- 子不调 `attach/detach`，**不** emit 自己的 `agent.run.started/completed`；
- 子即便声明了不同 model，LLM 调用实际走的是**父的** gateway（子复用父 ioPort，其 inner
  `DefaultIOPort` 包的是父 gateway）——一个连带的隐 bug。

### 关键事实：sub-agent 的 output 被记成父的 tool 结果

子 agent tool 的 handler 在父的 `ioPort.invokeTool(agentId, {goal,input}, execute)` 内执行，
返回值是子 `AgentResult.output`。于是 record 时父流里产生一对
`tool.requested` / `tool.responded`，**`tool.responded.output` 即子的最终 output**。

而 `ReplayingIOPort.invokeTool` 从 cache 按 `hashToolCall` 吐 output、**不调 execute()**
（`ReplayingIOPort.ts:38-55`）。因此父 replay 到子 tool 时**不会重跑子**——子的 output 直接从
父 cache 取。

由此推断（机制层面，当前无测试覆盖）：**带 sub-agent 的 run 今天 replay 是坏的**——子内部
llm/clock/uuid 事件在父流里、而父 replay 短路了子 tool，这些事件无人消费 → tail under-consume
→ `ReplayDivergenceError`。`Replay.test.ts` 9 个用例无一带 sub-agent，故未被撞到。这正是本
issue 的核心要修的。

## Scope 决策（重要）：模型 I 而非递归 replay

#47 正文 Scope 第 4 条与验收第 4 条字面写的是「`Milkie.replay` 沿 spawn 树**递归下钻**」。
本设计**不采用递归**，改用「子是不透明 tool 结果 + 每条 run 独立可 replay」的模型 I。

两个模型：

- **模型 I（采用）**：父 replay 到子 tool 时照常从父 cache 吐录好的 output、不重跑子。子 I/O
  移入 `<childRunId>.jsonl` 后，父 cache 里子相关只剩那一条 `tool.responded` → 父 replay 干净
  消费。要看/replay 子，单独 `Milkie.replay(childRunId)` 重放子文件。「nested sub-trace」退化
  为导航/视图概念（顺 `agent.spawned.childRunId` 加载、按需独立 replay）。
- **模型 II（否决）**：父 replay 重新 spawn 子、子从自己 cache 跑。代价：(1) **双真相**——子
  output 既是父 cache 的 `tool.responded` 又是重跑子的产物，二者必须永远相等，且为强制递归还得
  把父那条 `tool.responded` 排除出 cache（连 record 侧也要改）；(2) **吞错捕获**——子 `run()`
  把异常吞成 `status:'error'`（`AgentRuntime.ts:793-799`），子的 `ReplayDivergenceError` 不会
  上抛，须给每个子 port 套 divergence-capture proxy 并让其穿过整棵树冒到顶层重抛。

模型 II 的唯一收益是 **fork**（父 replay 到某点后让子活着跑出不同结果）。fork 在 #47 是
Related/Blocks 的未来项，验收只要求「replay 结果一致」。模型 I 直接满足该意图——父的结果完全由
其 cache 里那条子 output 决定，子的内部确定性对父无关紧要——且避免了双真相与吞错捕获两个根
问题。故：**收益不明显的递归不做。**

**与 #47 字面的偏离需记账**：本设计满足 #47 验收的*意图*（replay 一致 + 子有独立 sub-trace +
子独立可 replay），但机制上不递归。建议**更新 #47 正文**：把「递归下钻」改为「子独立可 replay +
导航跟随」，并把递归 replay 归入未来 fork issue（fork 时双真相会被正面解决，那才是它该被解决的
地方）。

## 设计

### 1. Record 侧（本 issue 主体）

**(a) 注入 per-child port 工厂**（仿现有 `childRecorderFactory`，Milkie → AgentRuntime）：

```ts
makeChildPort?: (childRunId: string, childConfig: AgentConfig) => IIOPort
```

Milkie 实现：

```ts
(childRunId, childConfig) =>
  new RecordingIOPort(
    new DefaultIOPort(this.gatewayOverride ?? createGateway(childConfig.model)),
    this.eventStore!,
    childRunId,
  )
```

`createGateway` / `RecordingIOPort` 留在 Milkie（它们本就住那），runtime 只拿一个 `IIOPort`
工厂 → 不违反 invariant #3（runtime 不依赖 Agent Trace）。子用自己的 model gateway，顺带修掉
「子走父 model」隐 bug。`gatewayOverride` 仍优先。

**(b) spawn 路径改写**（`makeSubAgentTool.handler` 内 + 构造子 runtime 的 `spawnFn`）：

- 子不再复用 `this.ioPort`，改用 `makeChildPort(childRunId, subConfig)` 铸的独立 port；
- 子 `AgentRuntime`：`agentRunId = childRunId`（独立）、`parentId = this.agentRunId`；
- 在子 `run()` 前后对子 port 调 `attach({parentId})` / `detach({status})` → 子 emit 自己的
  `agent.run.started{parentId}` / `agent.run.completed{status}` 进 `<childRunId>.jsonl`；
- 回退：`makeChildPort` 未注入时回退到 `this.ioPort`（replay 路径不注入；replay 也不进 handler，
  见 §2）。

attach/detach 是 `RecordingIOPort` 上的方法（非 `IIOPort` 契约）。在 spawn seam 用
`instanceof RecordingIOPort` 守卫调用，与 `Milkie.invoke`（`Milkie.ts:219`）同款处理。

**(c) childRunId 用直 `uuidv4()`（信息性，绕 cache）**。模型 I 下父 replay 不重跑子，childRunId
无需 replay 可复现；其可发现性由 `agent.spawned.childRunId` 保证。

**(d) 关键收口：父侧子边界 id 改信息性。** 今天 `childContextId` / `childTraceId` / `taskId`
走 `this.ioPort.uuid()`（`AgentRuntime.ts:217,218,220`）→ 进父 cache。模型 I 下 replay 短路子
tool、不进 handler → 这些 uuid 不被消费 → under-consume 散度。解法：三者改直 `uuidv4()`
（对父 replay 它们是信息性的；子独立 replay 用的是子文件里自己的 contextId）。**这是模型 I
自洽的前提**。注意 `batchId`（`AgentRuntime.ts:1009`，在 `invokeTool` 之前的 executeTools 里）
保持走 ioPort.uuid()——它在 handler 之外、replay 照常分配与消费，不受影响。

### 2. Replay 侧（近乎零改动）

`Milkie.replay(父runId)`：**基本不动**。子 tool 照常被 `ReplayingIOPort.invokeTool` 从父 cache
吐 output、不重跑子。配合 §1，父 cache 里子相关只剩那条 `tool.responded` → over/under-consume
皆通过。

子独立 replay：`Milkie.replay(childRunId)` 直接把 `<childRunId>.jsonl` 当顶层 run 重放——已有
机制即可（`extractRunSnapshot` 读子文件 `agent.run.started` 拿 goal/input/contextId/parentId）。

`parentId` 无需新增 schema：`AgentRunStartedPayload.parentId` 已是可选字段（`types.ts:87`）且
`extractRunSnapshot` 已读取（`RunSnapshot.ts:37`），顶层目前传 undefined。本 issue 只需在子的
`attach()` 时**填** `parentId = 父 runId`，满足验收「子 `agent.run.started.parentId` == 父
runId」。

### 3. 数据模型 + resume

- `ChildAgentRecord` 加 `runId` 字段（= childRunId），随父 checkpoint 的 `children[]` 持久化
  →给出子的**持久身份**，支撑 lineage 与 resume 续跑。
- `agent.spawned` / `agent.returned` 的 `childRunId` 值从 childContextId 升级为真 childRunId
  （只换值，schema 不变——#24 已留好）。

**resume 续跑（B 的归属）**：用户选 B（子延续同一 runId，理由：runId 是 run 的稳定身份，中断是
暂停非结束；不然 invariant #11「一个事件 + 一条 sub-trace」只对没被中断过的子成立；且与 fork
焊死）。本 issue **把身份做到位**（持久化 `runId`），但完整的「父 resume → 子按稳定键续跑同一
`<childRunId>.jsonl`」留明确 follow-up，理由：

- 当前无测试要求 resume 续跑（现有 supervisor 测试只覆盖中断传播 + 子 checkpoint 记录，
  `AgentRuntime.test.ts:375`）；
- resume 时父重新 spawn 要**匹配回**之前的 `ChildAgentRecord` 才能复用 runId，而当前
  `taskId`/`contextId` 每次 spawn 新生成、无稳定匹配键。完整续跑需引入稳定 spawn 键（如
  父 turn + 子 agentId + 调用序号），属独立工作量。

故本 issue：身份持久化（runId 进 record）+ 留 follow-up issue 做稳定键与续跑执行。

### 4. 错误处理

- 子 run 报错：父 `tool.responded` 记 error（已有路径，`RecordingIOPort.ts:184-204`）+ 父
  `agent.returned{status:'error'}`（#24 已有）；子文件含子自己的
  `agent.run.completed{status:'error'}`。
- 无新增吞错风险：模型 I 不重跑子，不存在「子 run() 吞 divergence」的捕获问题。
- 子 port 写入失败：沿用 `RecordingIOPort` best-effort 语义，不改子业务结果。

## 测试（`src/__tests__/`，沿用 StubGateway + sub-agent harness）

1. 父 spawn 子 → 子 llm/tool/clock/uuid 落 `<childRunId>.jsonl`；父流子相关只含 sub-agent 的
   `tool.requested/responded` + `agent.spawned/returned`（无子 llm 事件）；
2. 子文件含 `agent.run.started{parentId == 父runId}` / `agent.run.completed{status}`；
   `agent.spawned.childRunId == 子文件 runId == agent.returned.childRunId`；
3. **`Milkie.replay(父runId)` 结果与原 run 一致、无 divergence**（今天坏的，本 issue 转绿——
   核心验收）；
4. `Milkie.replay(childRunId)` 独立重放子、无 divergence；
5. 子声明的 model 与父不同时，子 LLM 走子自己的 gateway；
6. `ChildAgentRecord.runId` 持久化进父 checkpoint；
7. 现有单 run replay（`Replay.test.ts`）+ supervisor 中断/恢复（`AgentRuntime.test.ts`）全绿。

## 验收（本 issue 范围内）

- [ ] 子 agent 拥有独立 `agentRunId`，I/O 落 `<childRunId>.jsonl`
- [ ] 子 emit `agent.run.started{parentId == 父runId}` / `agent.run.completed{status}`
- [ ] `agent.spawned.childRunId` == 子 run 文件的 runId（值升级为真 childRunId）；
      `agent.returned.childRunId` 与之一致、status 与子 `agent.run.completed.status` 一致
- [ ] `Milkie.replay(父runId)` 能 replay 含 sub-agent 的 run，结果与原 run 一致、不
      over/under-consume（经模型 I：子 tool 由父 cache 吐 output；父侧子边界 id 信息性化）
- [ ] `Milkie.replay(childRunId)` 能独立重放子 run
- [ ] 子用自己 model 的 gateway（修连带 bug）
- [ ] `ChildAgentRecord.runId` 持久化
- [ ] 现有单 run replay + supervisor 中断/恢复 + skill 生命周期测试全绿

## 显式 deferral（不在本 issue）

- 递归 replay（模型 II）：否决，归入未来 fork issue；
- 父 resume → 子按稳定键续跑同一 `<childRunId>.jsonl` 的**执行**（本 issue 只做身份持久化）→
  新 follow-up（稳定 spawn 键 + 续跑）；
- fork（replay 到某点后子活着分叉）；
- `causedBy`（→ #30）。

## Related

- ARCH.md invariant #11、L17（runs addressable/replayable）、L402
- 依据：`docs/superpowers/specs/2026-05-29-agent-spawn-return-events-design.md`（#24，(1)/(2) 焊缝）
- Blocks: #27（sub-agent tree 跨 run 导航）；lineage 跨 run（#37 / #41）
- Follow-on: 父 resume→子续跑（稳定 spawn 键）；fork issue（递归 replay 在此落地）
