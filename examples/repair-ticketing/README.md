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

服务器使用 agent 自带模型（volcengine doubao，openai-compatible），运行前请在环境中配置
对应的 provider 凭据。端口可用 `PORT` 覆盖。

依次输入即可看到四个槽位 chip（站点 / 楼宇 / 部门 / 负责人）逐个点亮，最后给出故障描述后，
右侧生成工单卡片。例如：`总部` → `主楼` → `IT网络部` → `王芳` → `投影仪无法开机`。

## 它演示了什么（#175 轻量分档）

- **单态自主 agent**（见 `src/agent.ts` 的 `repair` 状态）：一个 `llm` 状态、四个工具
  （`lookup_entity` / `commit_entity` / `commit_description` / `assemble_ticket`），
  在同一个工具循环里跑完「逐级定位 → 描述 → 开单」。无 `on:` 边、无 `ctx.emit`、无业务事件。
  对应 `docs/design/175 §6` 的最小机制递增：
  - **软顺序 / 阶段聚焦** → `systemPrompt`（软引导，非硬拓扑）。
  - **槽位完整性** → tool param schema + 解析器自带校验（硬地板，本地规则）：
    `lookup_entity` 检索候选、`commit_entity` 按 `pinned` 上级分支校验后写入 WM。
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

> #175 注：去多态后，**已完成的 turn 不再持久化 WM**（跨 turn 槽填充经 `need_input → paused`
> 是后续切片）。因此整个「定位 → 描述 → 开单」在**一个 `invoke` turn** 内跑完——`pinned`
> 上级过滤让解析器从同一句话里逐级消歧。e2e 据此用单 turn 驱动全流程。

## 测试

```bash
./node_modules/.bin/jest examples/repair-ticketing
```

`repair-ticketing.e2e.test.ts` 用确定性 stub gateway 在单态、单 turn 内跑通全流程（happy path），
断言确定性工单；并单测 **action precondition**（未集齐拒绝开单、缺字段列表、不抛错可恢复），
以及覆盖解析器的各类返回状态（别名命中 / 缺级 / 歧义 / unknown / invalid_selection / corrected 回退）。

## 不在范围内

- 不改 `resolver/EntityResolver.ts`（#167 的产物）。
  注：`src/tools/entity-resolver.ts`（#166）本切片仅删掉已不存在的 `ctx.emit('SLOTS_COMPLETE')`
  —— runtime 去多态后 `ToolContext` 无 `emit`，槽完整性改由下游 `assemble_ticket` precondition 从 WM 读回。
- 跨 turn 槽填充（`need_input → paused` 挂起/恢复）—— #175 后续切片，本示例先单 turn。
- 无热更新、无鉴权、无真实工单系统对接。
