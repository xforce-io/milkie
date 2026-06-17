# repair-ticketing — 层级实体报修工单（S-011 Path D 的 Web 版）

一个最小可跑的报修工单助手：用户逐级说出报修对象（站点 → 楼宇 → 部门 → 负责人），
再用一句话描述故障，系统**确定性**生成结构化工单。它是 [S-011 Path D](../../docs/stories)
（多状态 FSM + 层级槽填充）的 Web UI 形态，把 #167 的可移植解析器内核接进 milkie 的
`ToolDefinition` 与 FSM 里在线运行——无子进程、无 CLI spawn。

## 运行

```bash
npm install          # 在仓库根目录
npx tsx examples/repair-ticketing/src/server.ts
# 打开 http://localhost:7979
```

服务器使用 agent 自带模型（volcengine doubao，openai-compatible），运行前请在环境中配置
对应的 provider 凭据。端口可用 `PORT` 覆盖。

依次输入即可看到四个槽位 chip（站点 / 楼宇 / 部门 / 负责人）逐个点亮，最后给出故障描述后，
右侧生成工单卡片。例如：`总部` → `主楼` → `IT网络部` → `王芳` → `投影仪无法开机`。

## 它演示了什么

- **三状态 FSM**（见 `src/agent.ts`）：
  - `collecting_entities`（`llm`）：`lookup_entity` + `commit_entity` 在线逐级解析四级实体，
    四级齐备时 emit `SLOTS_COMPLETE` → `collecting_description`。
  - `collecting_description`（`llm`）：仅一个**非 HER** 工具 `commit_description`，把用户的自由
    文本描述写入工作记忆并 emit `DESCRIPTION_READY` → `emit_ticket`。
  - `emit_ticket`（`action`）：纯函数 handler 从工作记忆**已确认字段**确定性拼装工单
    （`assembleTicket`，无 LLM），`DONE` → `completed`（终态）。
- **确定性工单**：最终工单不由 LLM 编写，而是从已验证的 WM 字段组装，因此可被测试精确断言
  （`src/__tests__/repair-ticketing.e2e.test.ts`）。
- **可移植内核**：`resolver/EntityResolver.ts`（#167）是纯库——不碰文件系统，JSON 进 / JSON 出。
  非 TS 调用方可经 CLI 包装器（`bin/entity-resolver`、`scripts/resolver.ts`）复用同一内核。

## HTTP / SSE 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET`  | `/` | 返回 `public/index.html` |
| `POST` | `/chat` | `{ input, contextId? }` → 跑一轮 → `{ contextId, runId, status, output, workingMemory, ticket }` |
| `GET`  | `/events/:contextId` | SSE，回放并实时推送 WM 变更事件（`wm.mutated`），驱动 chip 实时更新 |

工单从 `POST /chat` 的 **live 返回值**（`output` / `ticket`）读取，**不读 checkpoint**：
`emit_ticket → completed` 是终态 turn，按 #172 终态 turn 不落 checkpoint，读 checkpoint 会丢最终态。

## 测试

```bash
npx jest examples/repair-ticketing
```

`repair-ticketing.e2e.test.ts` 用确定性 stub gateway 跑通三状态全流程（happy path），断言确定性
工单，并覆盖解析器的各类返回状态（别名命中 / 缺级 / 歧义 / unknown / invalid_selection / corrected 回退）。

## LLM 鲁棒性与端到端质量评测（#174）

`eval/` 目录用**真实模型**（doubao-seed-2.0-lite，经 `createGateway`，无 stub gateway）度量 HER 槽填充
与端到端工单质量。打分**零 LLM judge**：ground-truth 是离散实体 id，精确匹配即可、且完全确定可复现。

```bash
# 需要 live 模型凭证（VOLCENGINE_TOKEN + VOLCENGINE_API_BASE）；缺凭证时打印 SKIP 并退出 0
npm run eval:repair
```

- **数据集** `eval/cases.jsonl`：覆盖 `canonical` / `colloquial` / `multi-level-in-one-turn` /
  `alias` / `typo` / `synonym` / `ambiguous` / `unknown` / `correction` / `cross-branch` 等标签，每标签 3–5 例。
- **harness** `eval/run-eval.ts`：复用 #162 导出的 `repairTicketingAgentConfig` + `buildRepairTicketingTools`，
  逐轮 `milkie.invoke`（同一 `contextId`）；槽位 id 从 `wm.mutated` 事件读取、工单从 live 返回值读取，**均不读 checkpoint**
  （#172：终态 turn 不落 checkpoint）。
- **打分** `eval/scoring.ts`（纯函数，含单测 `eval/__tests__/scoring.test.ts`，无凭证即可在 CI 跑）：
  4 槽全匹配率 + 单槽率、工单字段精确匹配率、澄清行为准确率（ambiguous/unknown）、纠错成功率、
  平均轮数与失败归因分布（wrong level / hallucinated id / premature emit / missing slot / missing ticket）。
- **报告**：写入 `eval/reports/`（已 gitignore）。

`eval:repair` 挂在 `npm run test:e2e:live` 之下，默认 CI 不跑、需 live 凭证。

### 基线（baseline）

首次 live run 的汇总指标回填于下表，并同步到 issue #174。指标口径见 `eval/scoring.ts`，
表格列与 `eval/run-eval.ts` 的 `renderReport` 输出一一对应；复跑只需：

```bash
VOLCENGINE_TOKEN=… VOLCENGINE_API_BASE=… npm run eval:repair
# 报告写入 eval/reports/eval-<时间戳>.{md,json}（已 gitignore），把 md 的汇总指标回填到下表
```

| 指标 | 数值 |
| --- | --- |
| model | _待 live run 回填_ |
| 用例数 | _待 live run 回填_ |
| 整体通过率 | _待 live run 回填_ |
| 平均轮数 | _待 live run 回填_ |
| 4 槽全匹配率 | _待 live run 回填_ |
| · site / building / department / assignee 单槽率 | _待 live run 回填_ |
| 工单字段精确匹配率 | _待 live run 回填_ |
| 澄清行为准确率（ambiguous/unknown） | _待 live run 回填_ |
| 纠错成功率（correction） | _待 live run 回填_ |
| 失败归因分布 | _待 live run 回填_ |

> 注：本次 patch 的执行环境已配置 `VOLCENGINE_TOKEN` + `VOLCENGINE_API_BASE`，但沙箱禁用了
> `node`/`npm`/`tsx` 进程执行，无法在此跑出 live 数字（拒绝在基线中填入伪造数据）。请在可执行 shell 中
> 跑一次上面的命令，将 `eval/reports/` 生成的 md 汇总回填到本表与 issue #174。

## 不在范围内

- 不改 `resolver/EntityResolver.ts` 与 `src/tools/entity-resolver.ts`（#167 / #166 的产物）。
- 无热更新、无鉴权、无真实工单系统对接。
