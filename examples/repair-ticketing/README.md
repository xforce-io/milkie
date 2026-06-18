# repair-ticketing — 层级实体报修工单（S-011 Path D 的 Web 版）

一个最小可跑的报修工单助手：用户说出报修对象（站点 → 楼宇 → 部门 → 负责人），
再用一句话描述故障，系统**确定性**生成结构化工单。它是 [S-011 Path D](../../docs/stories)
（层级槽填充）的 Web UI 形态，把 #167 的可移植解析器内核接进 milkie 的
`ToolDefinition` 里在线运行——无子进程、无 CLI spawn。

> **#175 轻量分档迁移**：本示例原本用「三状态业务 FSM」
> （`collecting_entities → collecting_description → emit_ticket`，靠 `ctx.emit` 驱动转移）。
> #175 去多态后，core 不再做「用户态→用户态」业务转移，本示例改为
> **单态自主 agent + slot-filling + action precondition**（见 `docs/design/175-decore-multistate-fsm.md` §6）：
> 整个流程在**一个 `llm` 状态的工具循环**里跑完，无 `on:`、无业务事件。

## 运行

```bash
npm install          # 在仓库根目录
npx tsx examples/repair-ticketing/src/server.ts
# 打开 http://localhost:7979
```

服务器默认使用 agent 自带模型（volcengine doubao-seed-2.0-lite，openai-compatible）；若设置了
`DEEPSEEK_API_KEY`，则改用 DeepSeek 运行**同一个 agent**（与 `eval/run-eval.ts` 同口径，committed
默认不变）。运行前请配置对应 provider 凭据。端口可用 `PORT` 覆盖。

一句话说全可以，分多条消息**逐步补充**也可以——同一会话（`contextId`）内槽位会**跨轮累积**：
例如先发 `东楼`、再发 `张伟`，系统会据上文把张伟唯一定到「东楼·安全部」（E012）。右侧四个槽位 chip
（站点 / 楼宇 / 部门 / 负责人）随之逐个点亮，最后生成工单卡片。一句话直发亦可，含跳级：
`总部网络部王芳，投影仪坏了`（楼宇自动倒推补全）。

> 会话状态在进程内存（`MemoryStore`）：活跃会话内的多轮补充正常累积，但**服务器重启即丢**；
> 且暂无 durable 的挂起/恢复（`need_input → paused` 是后续切片）。这与「终态 turn 不落 checkpoint」
> （#172）是两回事——前者是进程内会话连续，后者是不写持久化检查点。

## 它演示了什么（#175 轻量分档）

- **单态自主 agent**（见 `src/agent.ts` 的 `repair` 状态）：一个 `llm` 状态、四个工具
  （`resolve_entities` / `commit_entity` / `commit_description` / `assemble_ticket`；`lookup_entity`
  仍注册供适配器契约/测试用，但已被 `resolve_entities` 取代，不挂在 agent 上），
  在同一个工具循环里跑完「逐级定位 → 描述 → 开单」。无 `on:` 边、无 `ctx.emit`、无业务事件。
  对应 `docs/design/175 §6` 的最小机制递增：
  - **软顺序 / 阶段聚焦** → `systemPrompt`（软引导，非硬拓扑）+「完成判定」引导（四级与描述齐了立即开单，不停在解析完）。
  - **槽位完整性** → tool param schema + 解析器自带校验（硬地板，本地规则）：
    `resolve_entities` 用本轮原话做融合召回、把【唯一确定】的层级直接确认；`commit_entity` 按 `pinned`
    上级分支校验后写入 WM——LLM 只在歧义时从**真实候选**里挑，绝不臆造 id。
  - **确定性倒推补全（#185）** → `resolve_entities` 看所有剩余级：某深层级一旦唯一确定，其祖先链上被
    跳过的中间级一并**确定性回填**（「总部网络部王芳」自动补主楼，「研发部赵明」连站点+楼宇都倒推出来）；
    仍有分歧的层级则吐出每个候选的完整 `chain=[{level,id,label}]` 交给模型判断/反问。**纯确定性、不依赖
    模型倒推、不碰 #167 内核**（仅在 `resolver/recall.ts` 暴露 `ancestors`）。
  - **跨阶段记忆** → 工作记忆里的四级 id 与描述（一次 turn 内）。
  - **一两条硬门（Y 成立才能 X）** → **action precondition**：`assemble_ticket`
    **未集齐四级实体与描述就拒绝开单**（"未报价不得维修" 的等价物），返回
    `preconditionFailed` + `missing` 列表让模型继续补齐，而不是抛错让整轮失败。
- **确定性工单**：工单不由 LLM 编写，而是 `assemble_ticket` 从已验证的 WM 字段纯函数组装
  （`assembleTicket`，无 LLM），因此可被测试精确断言（`src/__tests__/repair-ticketing.e2e.test.ts`）。
- **可移植内核**：`resolver/EntityResolver.ts`（#167）是纯库——不碰文件系统，JSON 进 / JSON 出。
  非 TS 调用方可经 CLI 包装器（`bin/entity-resolver`、`scripts/resolver.ts`）复用同一内核。

## HTTP / SSE 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET`  | `/` | 返回 `public/index.html` |
| `POST` | `/chat` | `{ input, contextId? }` → 跑一轮 → `{ contextId, runId, status, output, workingMemory, ticket }` |
| `GET`  | `/events/:contextId` | SSE，回放并实时推送 WM 变更事件（`wm.mutated`），驱动 chip 实时更新 |

工单从 `POST /chat` 的 **live 返回值**（`output` / `ticket`）读取，**不读 checkpoint**：
`assemble_ticket` 返回工单 JSON，模型把它作为本轮最终文本原样回复，所以 live `output` 即工单。

> #172/#175 注：终态 turn **不写 durable checkpoint**（#172），且暂无正式的挂起/恢复
> （`need_input → paused` 是后续切片）。但同一会话（`contextId`）内 WM 仍由进程内 `MemoryStore`
> **跨轮保留**，多轮逐级补充会正常累积（服务器重启即丢）。e2e/happy-path 选择在**一个 `invoke` turn**
> 内跑完，是为了断言确定，而非跨 turn 不可行——`pinned` 上级过滤让解析器从同一句话里逐级消歧。

## 测试

```bash
./node_modules/.bin/jest examples/repair-ticketing
```

`repair-ticketing.e2e.test.ts` 用确定性 stub gateway 在单态、单 turn 内跑通全流程（happy path），
断言确定性工单；并单测 **action precondition**（未集齐拒绝开单、缺字段列表、不抛错可恢复），
以及覆盖解析器的各类返回状态（别名命中 / 缺级 / 歧义 / unknown / invalid_selection / corrected 回退）。

**纠错（correction）作为功能测试**：用户撤回改选（「不对，应该是网络部」）的**机制**是确定性的，
故沉淀为功能测试而非 eval——分两层：① **同级覆盖**（`commit_entity` 重提交一个已填层级 → 覆盖旧 id 并报
`corrected`，区别于上面那条「祖先冲突自动纠错」）；② **流程中途纠错的端到端传播**（stub gateway 先提交错的
部门 D02、再纠正为 D03，断言最终工单带 D03 而非残留 D02）。至于「真实模型会不会主动撤回」属模型行为、
且天然多轮，留给未来的多轮对话 eval（见下）。

## LLM 鲁棒性与端到端质量评测（#174）

`eval/` 目录用**真实模型**（doubao-seed-2.0-lite，经 `createGateway`，无 stub gateway）度量 HER 槽填充
与端到端工单质量。打分**零 LLM judge**：ground-truth 是离散实体 id，精确匹配即可、且完全确定可复现。

> **聚焦单轮 HER 解析**：本 eval 只考「**一句残缺/模糊的话 → 解析出完整 HER**」——每个用例就是一轮输入，
> 期望要么解析出完整四级（`slots`/`ticket`），要么在信息不足时**追问/拒绝**（`outcome:"clarify"/"reject"`，
> 不许臆造）。跨轮状态承接（`need_input→paused` 挂起/恢复）、纠错撤回等**多轮对话行为**是另一条正交的功能轴，
> 不在此 eval 内：确定性的部分已下沉为功能测试（见上文「纠错作为功能测试」），其余待后续**多轮对话 eval**。
> 单轮聚焦把 HER 解析质量与对话状态承接这两条正交的轴分开，避免多轮的状态管理掺进来污染归因
> （多轮补充本身是 work 的；durable 挂起/恢复 `need_input→paused` 才是尚未落地的部分）。

```bash
# 需要 live 模型凭证（VOLCENGINE_TOKEN + VOLCENGINE_API_BASE）；缺凭证时打印 SKIP 并退出 0
npm run eval:repair
```

- **数据集** `eval/cases.jsonl`（**全部单轮**，每条一句话）：覆盖 `canonical`（完整路径无降级）/ `colloquial`（口语）/
  `alias`（别名）/ `typo`（错字）/ `synonym`（同义）/ `cross-branch`（重名靠上下文消歧）/ `skip-level`（跳级）/
  `ambiguous`（歧义→追问）/ `unknown`（不存在→拒绝）等标签；出工单的用例额外带 `oneshot`（顺带考 `descriptionCleanOk`：
  把整句原话之外的故障子串干净存入 description，不让层级原话泄漏）。
  - `skip-level`：用户**跳过中间层级**（如「总部网络部王芳」漏掉楼宇，甚至「研发部赵明」连站点都不给）。
    `resolve_entities` 会**确定性倒推补全**——某深层级唯一确定时，其祖先链上被跳过的级一并回填（词典里
    「网络部」只在主楼，故楼宇可定），期望补全四级（写 `slots`/`ticket`）；倒推后仍跨分支歧义的
    （如「张伟」横跨网络部/安全部）期望**追问**（`outcome:"clarify"`，下游级留空不许猜）。
- **harness** `eval/run-eval.ts`：复用 #162 导出的 `repairTicketingAgentConfig` + `buildRepairTicketingTools`，
  每个用例一轮 `milkie.invoke`；槽位 id 从 `wm.mutated` 事件读取、工单从 live 返回值读取，**均不读 checkpoint**
  （#172：终态 turn 不落 checkpoint）。
- **打分** `eval/scoring.ts`（纯函数，含单测 `eval/__tests__/scoring.test.ts`，无凭证即可在 CI 跑）：
  通过与否**只由离散 id 决定**（4 槽全匹配 +（出工单时）工单字段精确匹配）；另报 4 槽全匹配率 + 单槽率、
  工单字段精确匹配率、澄清/拒绝行为准确率（ambiguous/unknown）、平均轮数与失败归因分布
  （wrong level / hallucinated id / premature emit / missing slot / missing ticket）。
  - **description clean 是独立软指标，不计入 pass**：oneshot 用例额外报「description 是否为干净故障子串（未泄漏层级原话）」，
    但描述脏**不判 HER 解析失败**——自由文本抽取质量与层级解析是两个正交的轴，混在一起会让召回数字失真。
  - `correction success` 指标随 correction 用例下沉为功能测试后已从 `renderReport` 移除。
- **报告**：写入 `eval/reports/`（已 gitignore）。

`eval:repair` 挂在 `npm run test:e2e:live` 之下，默认 CI 不跑、需 live 凭证。

### 基线（baseline）

下表为 live run 的汇总指标。指标口径见 `eval/scoring.ts`，列与 `eval/run-eval.ts` 的 `renderReport`
输出一一对应。复跑：

```bash
# doubao（committed 默认）：
VOLCENGINE_TOKEN=… VOLCENGINE_API_BASE=… npm run eval:repair
# 或 DeepSeek（下表所用模型）：
DEEPSEEK_API_KEY=… npm run eval:repair
# 报告写入 eval/reports/eval-<时间戳>.{md,json}（已 gitignore）
```

| 指标 | 数值 |
| --- | --- |
| model | `deepseek-chat` |
| 用例数 | 33（全单轮） |
| 整体通过率 | **100.0%** |
| 平均轮数 | 1.00 |
| 4 槽全匹配率 | 100.0% |
| · site / building / department / assignee 单槽率 | 100% / 100% / 100% / 100% |
| 工单字段精确匹配率 | 100.0% |
| 澄清/拒绝行为准确率（ambiguous/unknown） | 100.0%（8 例） |
| description clean（软指标，不计入 pass） | 100.0%（25 例） |
| 失败归因分布 | 无 |

> 注：上表为 **DeepSeek（`deepseek-chat`）** 的数字——committed 示例模型 doubao-seed-2.0-lite 偏弱
> （eval 注释亦如此说明），harness 凭 `DEEPSEEK_API_KEY` override 到 DeepSeek 跑出该结果，committed 默认未改。
> 演进：确定性倒推回填 + description 解耦把整体从 57.6% 拉到 90.9%（HER 槽位即到 100%），再加 systemPrompt
> 「完成判定」引导收掉残留的 missing_ticket → 100%。

## 不在范围内

- 不改 `resolver/EntityResolver.ts`（#167 的产物）。
  注：`src/tools/entity-resolver.ts`（#166）本切片仅删掉已不存在的 `ctx.emit('SLOTS_COMPLETE')`
  —— runtime 去多态后 `ToolContext` 无 `emit`，槽完整性改由下游 `assemble_ticket` precondition 从 WM 读回。
- 跨 turn 槽填充（`need_input → paused` 挂起/恢复）—— #175 后续切片，本示例先单 turn。
- 无热更新、无鉴权、无真实工单系统对接。
