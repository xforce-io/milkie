# #88 答案错因诊断 — 专职 diagnoser agent over Trace 投影(实现 spec)

**Issue:** #88(diagnosable;Parent #20);设计层定稿见 #88 正文,本 doc 补**实现约束 + 测试策略**。
**日期:** 2026-06-01
**分支:** `feat/88-diagnoser-agent`(基于 main)

## 1. 目标(承 #88)

现有诊断答"机制"(Why)与"源支撑"(Provenance),答不了"答案对不对 / 错在哪"。补一个**相关性链路断点**诊断:沿 `用户问题 → 工具 query → 命中证据 → 最终答案`,定位第一个与问题失配的跳。封装为**专职 diagnoser agent**(读 Trace 的旁观者),PoC 借住 `examples/agent-docs-qa`(归宿是 #89 的内置 agent 层,PoC 不阻塞于它)。

## 2. 组件与实现约束

```
examples/agent-docs-qa/
  agents/diagnoser.md       # diagnoser agent:方法论 systemPrompt + 声明 trace 工具
  tools/trace-tools.ts      # makeTraceTools(eventStore, objectStore) → 读-Trace 工具集
  server.ts                 # 多加载 diagnoser.md(再调一次 loadAgentFile)
  README.md                 # 标注 diagnoser 借住、归宿 #89
```

### 2.1 读-Trace 工具(确定性,可纯 TDD)

`makeTraceTools(eventStore, traceObjectStore)` 闭包返回(范式同 `makeCorpusTools`):
- `get_run_io({ runId })` → `{ question, finalAnswer }`(从 `agent.run.started.input` + `agent.run.completed.lastTextOutput`)。
- `get_execution({ runId })` → `ExecutionProjection`:`readByRunId(runId)` + hydrate regionContent(同 `trace report`/#70 endpoint 的方式)→ `buildExecutionProjection(events, { regionContent })`。给出步骤序列(LLM/工具调用、工具 query、命中证据、region 组成)。
- (可选,后置)`explain_step({ runId, eventId })` → 单步决策因果。

工具**只读、确定性**(包装核心投影,不解析原始事件)。这是诊断能力里**唯一能脱离 LLM 单测**的部分 —— 优先 TDD 它。

### 2.2 diagnoser agent(`diagnoser.md`)

- frontmatter 同 `sanguo-researcher.md`:`agentId: diagnoser`、单 llm state、`tools: [get_run_io, get_execution]`、model。
- systemPrompt = **诊断方法论**:拿 `targetRunId` → 调工具取 问题/答案/执行链 → 逐跳评估 relevance → 报告第一个 break + 理由。
- **输入** = `targetRunId`(`milkie.invoke('diagnoser', { input: targetRunId })`)。

### 2.3 输出契约(靠 prompt 约定 + parse,非强 schema)

milkie 无强结构化输出(只有 `responseFormat?` 透传)。故 systemPrompt **约定 diagnoser 的最终文本输出为 JSON**:
```json
{ "verdict": "ok" | "suspect",
  "firstBreak": { "step": "<eventId 或步序>", "what": "...", "why": "..." } | null,
  "explanation": "..." }
```
消费方 `JSON.parse`。PoC 接受"模型偶发不合格式"的脆性(prompt 里给严格格式要求 + 例子);强 schema enforcement 留后续(也可能推动核心加结构化输出能力)。

### 2.4 server 多 agent 加载

`server.ts:startServer` 现 `loadAgentFile(sanguo-researcher.md)` 单个 → 增加 `loadAgentFile(diagnoser.md)`(或遍历 `agents/`)。同一 milkie 实例两个 agent,靠 agentId 区分。

## 3. 测试策略(关键:诚实区分两层)

diagnoser 是 LLM agent,**测试分两层**:

- **确定性层(本 PoC 必做,TDD)**:
  - **读-Trace 工具单测**(纯函数级,易测):喂构造的 fixture run(events 数组),断言 `get_run_io` / `get_execution` 返回的投影正确(question/answer、步骤序列、工具 query、region 分组)。这是核心可确定性验证的部分。
  - **管道 + 输出契约集成测(stub LLM)**:stub gateway 让 diagnoser 调一次工具、输出一段预设 JSON;断言 (a) 工具被调用、拿到 targetRun 投影,(b) 最终输出能 `JSON.parse` 成契约结构。**stub 只证"管道通 + 契约成立",不证"判断对错"。**
- **真实 LLM 层(手动 / 可选 e2e,不进 CI 确定性套件)**:
  - 用真实 doubao,对"曹操爸爸 → grep 赤壁"这条真实 run 跑 diagnoser,**人工验证** `firstBreak` 指向"工具 query 与问题不相关"。这一层才验证诊断**质量**,但非确定,归 live e2e。

> 明确写下:stub 测不了诊断判断质量(那需要真实 LLM 推理)。PoC 的确定性保证落在"读-Trace 工具正确 + 管道通 + 输出契约成立";判断质量靠 live e2e 人工/抽样验证。

## 4. 边界

- 被诊断 run 与诊断 run 是**两个 runId**;diagnoser 自己跑也产生 Trace,别与被诊断 run 混。
- 判断逻辑在核心之外(diagnoser 是 example agent + LLM),核心只提供确定性投影。
- 不碰 event log 语义;读-Trace 工具只读。

## 5. 非目标(承 #88)

UI/panel 集成(endpoint + 按钮 + 渲染)、多轴诊断、grounding 复核(verifier 已有)、可注入 skill 形态、内置 agent 层机制(#89)。

## 6. 文件清单

| 文件 | 改动 |
|---|---|
| `examples/agent-docs-qa/tools/trace-tools.ts` | 新增 `makeTraceTools`(读-Trace 工具集) |
| `examples/agent-docs-qa/agents/diagnoser.md` | 新增 diagnoser agent(方法论 + 工具声明) |
| `examples/agent-docs-qa/server.ts` | 多加载 diagnoser.md |
| `examples/agent-docs-qa/__tests__/` | 读-Trace 工具单测 + stub 管道/契约测 |
| `examples/agent-docs-qa/README.md` | 标注 diagnoser 借住 / 归宿 #89 |
| (可选)live e2e | 真实 doubao 诊断"曹操爸爸→grep赤壁",人工验证 |
