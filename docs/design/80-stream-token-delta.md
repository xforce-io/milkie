# #80 — AgentRuntime 透传 token 级 message_delta（含 tool_call）

- Issue: #80
- Branch: `feat/80-stream-token-delta`
- 状态: 设计已批准（owner review 通过），进入实现
- 来源: alfred provider 替换调研。参考 **pi-ai**（`@mariozechner/pi-ai`，成熟 TS 多 provider 流式聚合）与 **dolphin**（alfred 现用、已验证的 `ToolCallsParser` 流式 tool_call 重建）。

## 1. 背景与约束

PoC 实测：经 sidecar 跑通对话后事件流只有 `llm.responded`（整段回复），**没有逐 token `message_delta`**。根因：

- `DefaultIOPort.invokeLLM()`（`src/runtime/IOPort.ts:62`）走 `gateway.complete()`（一次性）。
- `gateway.stream()` 已定义但是 dead code，没有任何 runtime 路径调用。
- `RecordingIOPort.invokeLLM`（`src/trace/RecordingIOPort.ts:150`）录制完整 `ModelResponse` 进 `llm.responded`；`CacheIndex` 按 `requestHash` 回放完整 response。

**最高约束（不可破坏）**：milkie 的字节级确定性 replay。录制/回放只认完整 `ModelResponse`。token delta 是**非确定的流式分片**，绝不能进 EventStore / CacheIndex。

## 2. 核心方案：双路-B（按需选路）

不采用「同时 `complete()` + `stream()` 两次调用」——2× 网络成本，且两次独立采样会导致**用户看到的 ≠ 实际提交进历史/被 replay 的**。

改为**同一次 LLM 调用按需选路**：

| 场景 | 路径 | 风险 |
|---|---|---|
| 有 UI 在看（web/telegram 对话） | `gateway.stream()` → 聚合 | 流重建 tool_call 的逻辑只在此跑 |
| 无人看（heartbeat/cron/子 agent/replay） | 现状 `gateway.complete()` | 零风险，行为不变 |

高风险的「从流重建 tool_call/usage」逻辑只在交互式 run 跑，爆炸半径受限。这契合 milkie「determinism first」。

## 3. 已批准的三项接口决策

1. **`onModelEvent` 作为选路信号 + delta 不入库**（不进 EventStore / CacheIndex），replay 零影响。
2. **不扩展 `ModelEvent` 类型**——现有联合类型已涵盖 `message_delta` / `tool_call_start` / `tool_call_delta` / `tool_call_done` / `usage`（`src/types/model.ts:32-38`），够用，只补 adapter 实现。
3. **`StreamAggregator` 放 `src/gateway/` 下独立模块**，与两个 adapter 同层。

## 4. 组件变更

### (a) 选路信号
`AgentInvokeRequest`（`src/types/common.ts:20`）/ `Milkie.invoke()` 新增可选：
```ts
onModelEvent?: (e: ModelEvent) => void
```
沿 `invoke → AgentRuntime（runLLMState）→ IOPort.invokeLLM` 传递。**有回调 = 流式路径，无回调 = `complete()`**。token delta 仅通过此回调出去。

> 注：`IIOPort.invokeLLM` 签名需加可选第二参 `onEvent?: (e: ModelEvent) => void`，三个实现（Default/Recording/Replaying）都要接住——但只有 Default 在 onEvent 存在时切流式；Recording 透传给 inner；Replaying 忽略（replay 无 onEvent）。

### (b) 新增 `StreamAggregator`（`src/gateway/StreamAggregator.ts`，~250 行）
- 输入：`AsyncIterable<ModelEvent>` + `onModelEvent` 回调。
- 行为：逐事件回调 `onModelEvent`（给 UI）；同时按 index（OpenAI）/ content_block index（Anthropic）累加 text 与 tool_call 分片。
- 输出：流末产出**与 `complete()` 字节等价的 `ModelResponse`**（content[] + toolCalls[] + usage + finishReason）。
- 借鉴 pi-ai：双键（index + id）容错聚合、partialArgs/partialJson 字符串累加后一次性 parse。

### (c) 补全两个 adapter 的 `stream()`（主要工作量）
- **`OpenAICompatibleAdapter.stream()`**（`src/gateway/OpenAICompatibleAdapter.ts:39`）：现状只 yield 文本 delta。需补：
  - `delta.tool_calls`：按 `index` 累加 `function.arguments`，id/name 取首片 → yield `tool_call_start` / `tool_call_delta` / `tool_call_done`。
  - `stream_options: { include_usage: true }` 拿 usage → yield `usage`。
  - 捕获 `finish_reason`。
- **`AnthropicAdapter.stream()` 的 `parseStreamEvent`**（`src/gateway/AnthropicAdapter.ts:147`）：现状只处理 `text_delta` + `output_tokens`。需补：
  - `content_block_start`(tool_use：id/name) → `tool_call_start`。
  - `content_block_delta`(input_json_delta：partial_json 拼接) → `tool_call_delta`。
  - `content_block_stop` → `tool_call_done`。
  - `message_start` / `message_delta`：完整 usage（input/output/cache_read/cache_creation）+ `stop_reason`。

### (d) IOPort 选路
`DefaultIOPort.invokeLLM(request, onEvent?)`：
```
onEvent 存在 → gateway.stream() → StreamAggregator(onEvent) → 完整 ModelResponse
onEvent 缺省 → gateway.complete()（现状）
```
**`RecordingIOPort` 录制逻辑不变**——它拿到的仍是完整 ModelResponse，照常录 `llm.responded`；只需把 onEvent 透传给 inner。replay cache 不变。

## 5. replay 安全论证

- 录的 `llm.responded.payload.response` 在两条路径下**结构等价**（聚合器保证 == complete() 结果）→ CacheIndex 不变 → replay 字节一致。
- delta 不写 EventStore → 不污染 CacheIndex / CausalCursor → replay 路径永不消费 delta。
- delta 事件**不经 IOPort 的 `uuid()` / `now()`** → 不破坏 record/replay 的 nondet 对称。
- replay 天然无 `onModelEvent` → `ReplayingIOPort.invokeLLM` 永走缓存 response，忽略 onEvent。

## 6. 测试策略

- **聚合器单测（核心）**：喂录制的分片 chunk 序列（借鉴 pi-ai faux/stub），断言聚合出的 ModelResponse == complete() 结果（text + tool_calls + usage + finishReason）。覆盖：纯文本、单 tool_call、并行多 tool_call、arguments 跨多片、中途 error。
- **adapter stream 单测**：mock OpenAI/Anthropic SDK 的流式 chunk，断言 yield 的 ModelEvent 序列正确。
- **determinism 回归**：现有 replay e2e（`test:e2e:deterministic`）必须全绿——证明 record/replay 不受影响。
- **选路等价测试**：同一 request，有/无 onModelEvent 分别走 stream/complete，最终 ModelResponse 等价。

## 7. 范围

- ✅ 一次到位含 tool_call 流式（文本 + 工具调用全程 delta）。
- ✅ OpenAI-compatible（volcengine/deepseek/openai）+ Anthropic 两个 adapter。
- ❌ 不改 RecordingIOPort / ReplayingIOPort / CacheIndex 的录制/回放语义（replay 零改动）。
- ❌ delta 不持久化（仅实时回调）；历史回放仍走事件日志的粗粒度事件。
- ❌ 不扩展 ModelEvent 类型。

## 8. 交付物链路

issue #80（一行摘要 + 本文链接）↔ 本设计文档 ↔ PR。
