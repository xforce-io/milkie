# #124 serve 端点增补:一次性 LLM + portable session

> SSOT。需求评估见 issue #124;前置能力见 #84(portable session 库)、#86(serve 最小版)。

## 1. 背景与原则

alfred 用 milkie 替换 dolphin provider,在 #86(serve 最小版)之上还需要两项跨进程能力:

1. **一次性 LLM**(对应 dolphin `call_llm`):memory/compressor 要直接发一枪 LLM 做摘要/压缩,**不跑完整 agent turn**。
2. **portable session 导出/导入**(对应 alfred 会话持久化):#84 已提供库能力,但 serve 没把它接成 HTTP 端点。

统一原则(与 #86 一脉相承):

> **能力在 Milkie 库层,serve 只补薄 HTTP 投影,CLI 命令暂不做。**

serve 与 CLI 是 Milkie 库的两个并列投影(传输不同、store 接线不同),谁都不"拥有"能力。`serve.ts` 的每个 handler 至今都只调 `milkie.*`、从不直接碰 gateway —— 本次新增保持这一不变量。

serve 当前**单 agent**(`serveMain` 只吃一个 `--agent`,`agentId` 在建 server 时钉死),且 **model 配置只存在于 agent 定义里**(`GatewayFactory` 从 agent 的 `model` 字段建 gateway,没有 server 级独立 model)。

## 2. 一次性 LLM

### 2.1 库层(实质改动)

新增 `Milkie.complete`,镜像既有 `DefaultIOPort.invokeLLM` 的语义,复用 `resolveGateway` / `resolveModel` / `aggregateStream`:

```ts
async complete(
  agentId: string,
  request: { system?: string; messages: Message[] },
  onEvent?: (e: ModelEvent) => void,
): Promise<ModelResponse>
```

- 用 `agentId` 解析该 agent 的 gateway 与 model(`config.model.model`),组装 `ModelRequest`。
- 无 `onEvent` → `gateway.complete(req)`(非流式)。
- 有 `onEvent` → `aggregateStream(gateway.stream(req), onEvent)`(逐 token 回调 + 聚合成 `ModelResponse`)。
- **不进 FSM、不产生 agent run、不写事件流**——纯一次性 completion。
- agent 不存在 → 抛错(与 `invoke`/`resume` 一致的报错文案)。

> 为何是一个带可选 `onEvent` 的方法而非 `complete`+`stream` 两个:这正是 milkie 既有的 I/O 抽象(`invokeLLM`)形态,避免重复。

### 2.2 serve 投影

```
POST /llm  { system?, messages, stream? }
```

- `messages` 必传(canonical `Message[]`,content 为数组);`system` 可选。
- 借**已加载的唯一 agent** 的 model 配置,**不传 agentId**(单 agent)。
- `stream` 省略 / false → JSON `{ output, usage? }`,`output` 为响应中 `text` 内容拼接。
- `stream` = true → `text/event-stream`:逐 `message_delta` 帧 + 终态 `done { usage }`;异常走 `error` 帧 + `done`(不裸断,语义对齐 `/chat`)。
- `messages` 缺失 → 400。

多 agent 是后续产品决策(#86 刻意单 agent);届时给 `/llm` 加**可选** `agentId` 即可,向后兼容,现在不预留(YAGNI)。

## 3. portable session 导出/导入

### 3.1 库层:已就绪(#84)

```ts
exportSession(contextId: string): Promise<PortableSession>
importSession(session: PortableSession): Promise<{ contextId: string }>
```

无需改动。

### 3.2 serve 投影(纯薄壳)

```
POST /session/export  { contextId }   → 200 PortableSession JSON
POST /session/import  { session }      → 200 { contextId }
```

- `/session/export`:`contextId` 缺失 → 400;无可导 session(库抛 `No session to export`)→ 404。
- `/session/import`:`session` 缺失 → 400;schemaVersion 不符(库抛 `Unsupported … schemaVersion`)→ 400。
- 库层抛错经 handler 映射为对应 4xx(默认 500 兜底)。

### 3.3 语义提醒(给 alfred,非本期实现)

export 是「最新 run + 子孙 run 的**前向状态快照**」,**非完整逐轮 transcript**(#84 决策 1)。若 compressor 需要全量历史原文,需补 by-context run 索引——单列 follow-up,本期不做。

## 4. 附带:落地孤儿提交

本地 `05c2169`(serve `/context/set·get·list` 端点 + 测试,#83 相关)此前从未合入 main,随本 PR 一并落地。

## 5. 不做

- #87 跨语言工具桥(P2):维持挂起,等 alfred 给出明确的 Python skill 复用清单再评估。
- CLI 的 `milkie llm` / `milkie session`:抽象已支持,YAGNI,有需求再加(~10 行投影)。

## 6. 测试(TDD)

库层(`milkie-complete.test.ts`):
- 非流式返回聚合 `output`;
- 流式逐 token 回调 `onEvent` 且聚合结果与非流式一致;
- agent 不存在抛错。

serve(`serve.test.ts` 增补):
- `/llm` 非流式返回 `{ output }`;
- `/llm` `stream=true` 收到 ≥2 个 `message_delta` + 终态 `done`;
- `/llm` 错误 gateway → `error` 帧 + `done`;
- `/llm` 缺 `messages` → 400;
- `/session/export` 在一次 `/chat` 后返回带 manifest/events/variables 的载荷;无 session → 404;
- `/session/import` 接受导出载荷返回 `{ contextId }`;schemaVersion 不符 → 400。
