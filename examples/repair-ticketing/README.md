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

## 不在范围内

- 不改 `resolver/EntityResolver.ts` 与 `src/tools/entity-resolver.ts`（#167 / #166 的产物）。
- 无热更新、无鉴权、无真实工单系统对接。
