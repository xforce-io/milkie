# #80 Token 级流式透传（含 tool_call）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让交互式 run 走 `gateway.stream()` 并把 token 级 `ModelEvent`（含 tool_call 分片）通过 `onModelEvent` 回调实时透传，同时聚合出与 `complete()` 字节等价的 `ModelResponse`；后台 run / replay 维持 `complete()`，RecordingIOPort/CacheIndex 零改动。

**Architecture:** 双路-B 按需选路。新增 `StreamAggregator`（消费 `AsyncIterable<ModelEvent>`，边回调边聚合）。补全两个 adapter 的 `stream()` 让其 yield tool_call 事件。`IIOPort.invokeLLM` 加可选 `onEvent` 参数：`DefaultIOPort` 在 onEvent 存在时走 stream+聚合，否则 complete()；`RecordingIOPort` 透传给 inner 不变录制；`ReplayingIOPort` 忽略。`Milkie.invoke` 加 `onModelEvent` 沿链传递。

**Tech Stack:** TypeScript, jest (ts-jest, ESM-style `.js` imports), OpenAI SDK, Anthropic SDK。设计文档：`docs/design/80-stream-token-delta.md`。

**测试运行约定：** `npx jest <path> -t '<name>'`（worktree 根目录）。全量单测：`npm run test:unit`；determinism 回归：`npm run test:e2e:deterministic`。

---

## File Structure

新建：
- `src/gateway/StreamAggregator.ts` — 消费 ModelEvent 流 → 聚合 ModelResponse + 逐事件回调（纯逻辑，不依赖 SDK）
- `src/__tests__/StreamAggregator.test.ts`
- `src/__tests__/OpenAICompatibleAdapter.stream.test.ts`
- `src/__tests__/AnthropicAdapter.stream.test.ts`
- `src/__tests__/AgentRuntime.stream.test.ts`

修改：
- `src/types/model.ts` — `IIOPort` 不在此；但 `ModelEvent` 已够用（不改类型）
- `src/runtime/IOPort.ts` — `IIOPort.invokeLLM` 加可选 `onEvent`；`DefaultIOPort` 选路
- `src/gateway/OpenAICompatibleAdapter.ts` — 补全 `stream()` 的 tool_call + usage + finish_reason
- `src/gateway/AnthropicAdapter.ts` — 补全 `parseStreamEvent` 的 tool_use + usage + stop_reason
- `src/trace/RecordingIOPort.ts` — `invokeLLM` 加 `onEvent` 透传给 inner
- `src/trace/ReplayingIOPort.ts` — `invokeLLM` 签名加 `onEvent`（忽略）
- `src/runtime/AgentRuntime.ts:962` — 调用 `ioPort.invokeLLM(request, this.onModelEvent)`；构造接收 `onModelEvent`
- `src/runtime/Milkie.ts` — `invoke()` 接 `onModelEvent`，传入 AgentRuntime
- `src/types/common.ts` — `AgentInvokeRequest` 加可选 `onModelEvent`

---

## Task 1: StreamAggregator — 纯文本聚合

**Files:**
- Create: `src/gateway/StreamAggregator.ts`
- Test: `src/__tests__/StreamAggregator.test.ts`

- [ ] **Step 1: 写失败测试（纯文本）**

```typescript
import { aggregateStream } from '../gateway/StreamAggregator'
import type { ModelEvent } from '../types/model'

async function* events(list: ModelEvent[]): AsyncIterable<ModelEvent> {
  for (const e of list) yield e
}

describe('StreamAggregator — text', () => {
  test('text deltas aggregate into one text content + forwards each event', async () => {
    const seen: ModelEvent[] = []
    const resp = await aggregateStream(
      events([
        { type: 'message_delta', data: { text: 'Hello' } },
        { type: 'message_delta', data: { text: ', world' } },
        { type: 'usage', data: { inputTokens: 10, outputTokens: 5 } },
      ]),
      (e) => seen.push(e),
    )
    expect(resp.content).toEqual([{ type: 'text', text: 'Hello, world' }])
    expect(resp.toolCalls).toEqual([])
    expect(resp.usage).toEqual({ inputTokens: 10, outputTokens: 5 })
    expect(seen).toHaveLength(3) // every event forwarded to the callback
  })
})
```

- [ ] **Step 2: 运行验证失败**

Run: `npx jest src/__tests__/StreamAggregator.test.ts -t 'text deltas'`
Expected: FAIL（`aggregateStream` 不存在）

- [ ] **Step 3: 写最小实现**

```typescript
import type { ModelEvent, ModelResponse, ModelUsage } from '../types/model.js'
import type { MessageContent } from '../types/common.js'
import type { ToolCall } from '../types/tool.js'

/**
 * Consume a provider event stream, forwarding every event to `onEvent`
 * (for live UI) while aggregating into a complete ModelResponse that is
 * byte-equivalent to what gateway.complete() would have produced.
 *
 * Tool-call argument fragments are accumulated by toolCallId as raw JSON
 * strings, then parsed once at the end (mirrors complete()'s JSON.parse).
 */
export async function aggregateStream(
  stream: AsyncIterable<ModelEvent>,
  onEvent?: (e: ModelEvent) => void,
): Promise<ModelResponse> {
  let text = ''
  let usage: ModelUsage | undefined
  let finishReason: string | undefined

  // tool calls accumulated in arrival order, keyed by toolCallId
  const toolOrder: string[] = []
  const toolById = new Map<string, { id: string; name: string; argsBuf: string }>()

  for await (const e of stream) {
    onEvent?.(e)
    switch (e.type) {
      case 'message_delta':
        text += e.data.text
        break
      case 'tool_call_start': {
        const id = e.data.toolCallId
        if (!toolById.has(id)) {
          toolById.set(id, { id, name: e.data.name, argsBuf: '' })
          toolOrder.push(id)
        }
        break
      }
      case 'tool_call_delta': {
        const t = toolById.get(e.data.toolCallId)
        if (t) t.argsBuf += typeof e.data.delta === 'string' ? e.data.delta : JSON.stringify(e.data.delta)
        break
      }
      case 'tool_call_done':
        // input already carried as parsed object by the adapter's done event;
        // prefer it when present (it's authoritative), else parse argsBuf later.
        {
          const t = toolById.get(e.data.toolCallId)
          if (t && e.data.input !== undefined) t.argsBuf = JSON.stringify(e.data.input)
        }
        break
      case 'usage':
        usage = e.data
        break
      case 'error':
        // surface as finishReason marker; aggregation still returns what we have
        finishReason = finishReason ?? 'error'
        break
    }
  }

  const content: MessageContent[] = []
  if (text) content.push({ type: 'text', text })

  const toolCalls: ToolCall[] = []
  for (const id of toolOrder) {
    const t = toolById.get(id)!
    let input: unknown
    try {
      input = t.argsBuf ? JSON.parse(t.argsBuf) : {}
    } catch {
      input = {}
    }
    content.push({ type: 'tool_use', id: t.id, name: t.name, input })
    toolCalls.push({ id: t.id, name: t.name, input })
  }

  return {
    content,
    toolCalls,
    ...(usage ? { usage } : {}),
    ...(finishReason ? { finishReason } : {}),
  }
}
```

- [ ] **Step 4: 运行验证通过**

Run: `npx jest src/__tests__/StreamAggregator.test.ts -t 'text deltas'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gateway/StreamAggregator.ts src/__tests__/StreamAggregator.test.ts
git commit -m "feat(stream): StreamAggregator 纯文本聚合 + 事件透传 (#80)"
```

---

## Task 2: StreamAggregator — tool_call 聚合（含并行多 tool + 跨片 arguments）

**Files:**
- Test: `src/__tests__/StreamAggregator.test.ts`（追加）

- [ ] **Step 1: 追加失败测试**

```typescript
describe('StreamAggregator — tool calls', () => {
  test('single tool call: arguments split across deltas → parsed object', async () => {
    const resp = await aggregateStream(events([
      { type: 'tool_call_start', data: { toolCallId: 'c1', name: 'search' } },
      { type: 'tool_call_delta', data: { toolCallId: 'c1', delta: '{"q":"three ' } },
      { type: 'tool_call_delta', data: { toolCallId: 'c1', delta: 'kingdoms"}' } },
      { type: 'tool_call_done',  data: { toolCallId: 'c1', input: undefined } },
    ]))
    expect(resp.toolCalls).toEqual([{ id: 'c1', name: 'search', input: { q: 'three kingdoms' } }])
    expect(resp.content).toEqual([{ type: 'tool_use', id: 'c1', name: 'search', input: { q: 'three kingdoms' } }])
  })

  test('parallel tool calls preserve order and separate buffers', async () => {
    const resp = await aggregateStream(events([
      { type: 'tool_call_start', data: { toolCallId: 'a', name: 'read' } },
      { type: 'tool_call_start', data: { toolCallId: 'b', name: 'grep' } },
      { type: 'tool_call_delta', data: { toolCallId: 'a', delta: '{"f":"x"}' } },
      { type: 'tool_call_delta', data: { toolCallId: 'b', delta: '{"p":"y"}' } },
      { type: 'tool_call_done',  data: { toolCallId: 'a', input: undefined } },
      { type: 'tool_call_done',  data: { toolCallId: 'b', input: undefined } },
    ]))
    expect(resp.toolCalls).toEqual([
      { id: 'a', name: 'read', input: { f: 'x' } },
      { id: 'b', name: 'grep', input: { p: 'y' } },
    ])
  })

  test('text + tool call coexist in content order (text first)', async () => {
    const resp = await aggregateStream(events([
      { type: 'message_delta', data: { text: 'let me search' } },
      { type: 'tool_call_start', data: { toolCallId: 'c1', name: 'search' } },
      { type: 'tool_call_delta', data: { toolCallId: 'c1', delta: '{}' } },
      { type: 'tool_call_done',  data: { toolCallId: 'c1', input: undefined } },
    ]))
    expect(resp.content).toEqual([
      { type: 'text', text: 'let me search' },
      { type: 'tool_use', id: 'c1', name: 'search', input: {} },
    ])
  })

  test('tool_call_done with explicit input overrides arg buffer', async () => {
    const resp = await aggregateStream(events([
      { type: 'tool_call_start', data: { toolCallId: 'c1', name: 't' } },
      { type: 'tool_call_delta', data: { toolCallId: 'c1', delta: '{"partial' } }, // malformed mid-stream
      { type: 'tool_call_done',  data: { toolCallId: 'c1', input: { ok: true } } },
    ]))
    expect(resp.toolCalls).toEqual([{ id: 'c1', name: 't', input: { ok: true } }])
  })
})
```

- [ ] **Step 2: 运行（Task 1 实现已覆盖这些路径，应直接通过）**

Run: `npx jest src/__tests__/StreamAggregator.test.ts`
Expected: PASS（全部）。若 `tool_call_done explicit input` 用例失败，检查 Task 1 实现中 `e.data.input !== undefined` 分支。

- [ ] **Step 3: Commit**

```bash
git add src/__tests__/StreamAggregator.test.ts
git commit -m "test(stream): StreamAggregator tool_call 聚合用例 (#80)"
```

---

## Task 3: OpenAI adapter stream() 补全 tool_call + usage + finish_reason

**Files:**
- Modify: `src/gateway/OpenAICompatibleAdapter.ts:39-53`
- Test: `src/__tests__/OpenAICompatibleAdapter.stream.test.ts`

OpenAI streaming chunk 的 `choices[0].delta.tool_calls[]` 每片含 `index`、首片含 `id`+`function.name`、`function.arguments` 跨片拼接。usage 需 `stream_options:{include_usage:true}`（最后一个 chunk 带 usage，choices 为空）。

- [ ] **Step 1: 写失败测试（mock SDK 流）**

```typescript
import { OpenAICompatibleAdapter } from '../gateway/OpenAICompatibleAdapter'
import type { ModelEvent } from '../types/model'

// Drive the private stream() with a fake OpenAI async stream by stubbing the
// client. We replace the adapter's `client.chat.completions.create`.
function adapterWithChunks(chunks: unknown[]): OpenAICompatibleAdapter {
  const adapter = new OpenAICompatibleAdapter({ apiKey: 'sk-test' })
  const fakeStream = (async function* () { for (const c of chunks) yield c })()
  ;(adapter as unknown as { client: { chat: { completions: { create: (...a: unknown[]) => unknown } } } })
    .client = { chat: { completions: { create: async () => fakeStream } } }
  return adapter
}

async function collect(adapter: OpenAICompatibleAdapter): Promise<ModelEvent[]> {
  const out: ModelEvent[] = []
  for await (const e of adapter.stream({ model: 'm', messages: [] })) out.push(e)
  return out
}

describe('OpenAICompatibleAdapter.stream — tool calls', () => {
  test('text + tool_call across chunks → events', async () => {
    const adapter = adapterWithChunks([
      { choices: [{ delta: { content: 'hi' } }] },
      { choices: [{ delta: { tool_calls: [{ index: 0, id: 'c1', function: { name: 'search', arguments: '{"q":' } } }] } }] },
      { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '"x"}' } }] } }] },
      { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
      { choices: [], usage: { prompt_tokens: 10, completion_tokens: 4 } },
    ])
    const events = await collect(adapter)
    expect(events).toContainEqual({ type: 'message_delta', data: { text: 'hi' } })
    expect(events).toContainEqual({ type: 'tool_call_start', data: { toolCallId: 'c1', name: 'search' } })
    expect(events).toContainEqual({ type: 'tool_call_delta', data: { toolCallId: 'c1', delta: '{"q":' } })
    expect(events).toContainEqual({ type: 'tool_call_delta', data: { toolCallId: 'c1', delta: '"x"}' } })
    expect(events).toContainEqual({ type: 'tool_call_done', data: { toolCallId: 'c1', input: { q: 'x' } } })
    expect(events).toContainEqual({ type: 'usage', data: { inputTokens: 10, outputTokens: 4 } })
  })
})
```

- [ ] **Step 2: 运行验证失败**

Run: `npx jest src/__tests__/OpenAICompatibleAdapter.stream.test.ts`
Expected: FAIL（只产出 message_delta，无 tool_call/usage 事件）

- [ ] **Step 3: 替换 stream() 实现**

把 `src/gateway/OpenAICompatibleAdapter.ts` 的 `stream()`（行 39-53）整体替换为：

```typescript
  async *stream(request: ModelRequest): AsyncIterable<ModelEvent> {
    const stream = await this.client.chat.completions.create({
      model:        request.model,
      messages:     this.convertMessages(request),
      tools:        request.tools?.map(t => ({
        type:     'function' as const,
        function: { name: t.name, description: t.description, parameters: t.inputSchema },
      })),
      tool_choice:  request.tools?.length ? 'auto' : undefined,
      stream:        true,
      stream_options: { include_usage: true },
    } as Parameters<typeof this.client.chat.completions.create>[0]) as unknown as AsyncIterable<{
      choices: Array<{ delta?: { content?: string; tool_calls?: Array<{ index: number; id?: string; function?: { name?: string; arguments?: string } }> }; finish_reason?: string }>
      usage?: { prompt_tokens: number; completion_tokens: number; prompt_tokens_details?: { cached_tokens?: number } }
    }>

    // toolCallId per OpenAI stream index; argsBuf accumulates for parse-on-done
    const byIndex = new Map<number, { id: string; argsBuf: string }>()

    for await (const chunk of stream) {
      const choice = chunk.choices?.[0]
      if (choice?.delta?.content) {
        yield { type: 'message_delta', data: { text: choice.delta.content } }
      }
      for (const tc of choice?.delta?.tool_calls ?? []) {
        let slot = byIndex.get(tc.index)
        if (!slot) {
          slot = { id: tc.id ?? `idx-${tc.index}`, argsBuf: '' }
          byIndex.set(tc.index, slot)
          yield { type: 'tool_call_start', data: { toolCallId: slot.id, name: tc.function?.name ?? '' } }
        }
        if (tc.function?.arguments) {
          slot.argsBuf += tc.function.arguments
          yield { type: 'tool_call_delta', data: { toolCallId: slot.id, delta: tc.function.arguments } }
        }
      }
      if (choice?.finish_reason && byIndex.size > 0) {
        for (const slot of byIndex.values()) {
          let input: unknown
          try { input = slot.argsBuf ? JSON.parse(slot.argsBuf) : {} } catch { input = {} }
          yield { type: 'tool_call_done', data: { toolCallId: slot.id, input } }
        }
        byIndex.clear()
      }
      if (chunk.usage) {
        const cached = chunk.usage.prompt_tokens_details?.cached_tokens
        yield {
          type: 'usage',
          data: {
            inputTokens:  chunk.usage.prompt_tokens,
            outputTokens: chunk.usage.completion_tokens,
            ...(cached !== undefined ? { cacheReadTokens: cached } : {}),
          },
        }
      }
    }
  }
```

- [ ] **Step 4: 运行验证通过**

Run: `npx jest src/__tests__/OpenAICompatibleAdapter.stream.test.ts`
Expected: PASS

- [ ] **Step 5: 跑现有 adapter 测试确保未回归**

Run: `npx jest src/__tests__/OpenAICompatibleAdapter.test.ts`
Expected: PASS（parseResponse 路径未动）

- [ ] **Step 6: Commit**

```bash
git add src/gateway/OpenAICompatibleAdapter.ts src/__tests__/OpenAICompatibleAdapter.stream.test.ts
git commit -m "feat(stream): OpenAI adapter stream() 补全 tool_call/usage/finish (#80)"
```

---

## Task 4: Anthropic adapter stream() 补全 tool_use + usage + stop_reason

**Files:**
- Modify: `src/gateway/AnthropicAdapter.ts:147-160`（`parseStreamEvent`）
- Test: `src/__tests__/AnthropicAdapter.stream.test.ts`

Anthropic 流式序列：`message_start`(usage) → `content_block_start`(tool_use: id/name) → `content_block_delta`(input_json_delta: partial_json) → `content_block_stop` → `message_delta`(stop_reason + usage)。

- [ ] **Step 1: 写失败测试（直接喂事件给 parseStreamEvent）**

```typescript
import { AnthropicAdapter } from '../gateway/AnthropicAdapter'
import type { ModelEvent } from '../types/model'

// parseStreamEvent is private generator over raw Anthropic events.
function parse(adapter: AnthropicAdapter, raw: unknown): ModelEvent[] {
  const gen = (adapter as unknown as { parseStreamEvent(e: unknown): Iterable<ModelEvent> }).parseStreamEvent(raw)
  return [...gen]
}

describe('AnthropicAdapter.parseStreamEvent — tool_use', () => {
  const adapter = new AnthropicAdapter({ apiKey: 'sk-test' })

  test('text_delta → message_delta', () => {
    expect(parse(adapter, { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'hi' } }))
      .toEqual([{ type: 'message_delta', data: { text: 'hi' } }])
  })

  test('tool_use block start → tool_call_start', () => {
    expect(parse(adapter, { type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: 't1', name: 'search' } }))
      .toEqual([{ type: 'tool_call_start', data: { toolCallId: 't1', name: 'search' } }])
  })

  test('input_json_delta → tool_call_delta (keyed by block index)', () => {
    // start establishes index→id mapping first
    parse(adapter, { type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: 't1', name: 'search' } })
    expect(parse(adapter, { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '{"q":"x"}' } }))
      .toEqual([{ type: 'tool_call_delta', data: { toolCallId: 't1', delta: '{"q":"x"}' } }])
  })

  test('content_block_stop for a tool block → tool_call_done with parsed input', () => {
    parse(adapter, { type: 'content_block_start', index: 2, content_block: { type: 'tool_use', id: 't2', name: 'read' } })
    parse(adapter, { type: 'content_block_delta', index: 2, delta: { type: 'input_json_delta', partial_json: '{"f":"a"}' } })
    expect(parse(adapter, { type: 'content_block_stop', index: 2 }))
      .toEqual([{ type: 'tool_call_done', data: { toolCallId: 't2', input: { f: 'a' } } }])
  })

  test('message_delta → usage + (no event for stop_reason alone)', () => {
    expect(parse(adapter, { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 7 } }))
      .toEqual([{ type: 'usage', data: { inputTokens: 0, outputTokens: 7 } }])
  })
})
```

> 注：parseStreamEvent 需要在实例上维护 `index → {id, buf}` 跨事件状态。把状态存为实例字段，并在 `content_block_stop` 后清理该 index。测试里同一 adapter 实例顺序喂事件即可。

- [ ] **Step 2: 运行验证失败**

Run: `npx jest src/__tests__/AnthropicAdapter.stream.test.ts`
Expected: FAIL（现状只处理 text_delta + output_tokens）

- [ ] **Step 3: 替换 parseStreamEvent + 加实例状态字段**

在 `AnthropicAdapter` 类体内（`client` 字段下方）新增状态字段：

```typescript
  // streaming: maps Anthropic content_block index → accumulating tool call
  private streamTools: Map<number, { id: string; buf: string }> = new Map()
```

把 `parseStreamEvent`（行 147-160）整体替换为：

```typescript
  private *parseStreamEvent(event: unknown): Iterable<ModelEvent> {
    const e = event as {
      type: string
      index?: number
      content_block?: { type: string; id?: string; name?: string }
      delta?: { type?: string; text?: string; partial_json?: string; stop_reason?: string }
      usage?: { output_tokens?: number }
    }

    if (e.type === 'content_block_start' && e.content_block?.type === 'tool_use' && e.index !== undefined) {
      this.streamTools.set(e.index, { id: e.content_block.id ?? '', buf: '' })
      yield { type: 'tool_call_start', data: { toolCallId: e.content_block.id ?? '', name: e.content_block.name ?? '' } }
      return
    }

    if (e.type === 'content_block_delta' && e.index !== undefined) {
      if (e.delta?.type === 'text_delta' && e.delta.text) {
        yield { type: 'message_delta', data: { text: e.delta.text } }
        return
      }
      if (e.delta?.type === 'input_json_delta' && e.delta.partial_json !== undefined) {
        const slot = this.streamTools.get(e.index)
        if (slot) {
          slot.buf += e.delta.partial_json
          yield { type: 'tool_call_delta', data: { toolCallId: slot.id, delta: e.delta.partial_json } }
        }
        return
      }
    }

    if (e.type === 'content_block_stop' && e.index !== undefined) {
      const slot = this.streamTools.get(e.index)
      if (slot) {
        let input: unknown
        try { input = slot.buf ? JSON.parse(slot.buf) : {} } catch { input = {} }
        this.streamTools.delete(e.index)
        yield { type: 'tool_call_done', data: { toolCallId: slot.id, input } }
      }
      return
    }

    if (e.type === 'message_delta' && e.usage) {
      yield { type: 'usage', data: { inputTokens: 0, outputTokens: e.usage.output_tokens ?? 0 } }
    }
  }
```

- [ ] **Step 4: 运行验证通过**

Run: `npx jest src/__tests__/AnthropicAdapter.stream.test.ts`
Expected: PASS

- [ ] **Step 5: 跑现有 Anthropic 测试确保未回归**

Run: `npx jest src/__tests__/AnthropicAdapter.test.ts`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/gateway/AnthropicAdapter.ts src/__tests__/AnthropicAdapter.stream.test.ts
git commit -m "feat(stream): Anthropic adapter stream() 补全 tool_use/usage/stop (#80)"
```

---

## Task 5: IIOPort.invokeLLM 加可选 onEvent；DefaultIOPort 选路

**Files:**
- Modify: `src/runtime/IOPort.ts`
- Test: `src/__tests__/StreamAggregator.test.ts`（追加 DefaultIOPort 选路用例）或新建 `src/__tests__/DefaultIOPort.routing.test.ts`

- [ ] **Step 1: 写失败测试（DefaultIOPort：有 onEvent 走 stream，无 onEvent 走 complete）**

新建 `src/__tests__/DefaultIOPort.routing.test.ts`：

```typescript
import { DefaultIOPort } from '../runtime/IOPort'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'

class SpyGateway implements IModelGateway {
  completeCalled = false
  streamCalled = false
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.completeCalled = true
    return { content: [{ type: 'text', text: 'from-complete' }], toolCalls: [] }
  }
  async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
    this.streamCalled = true
    yield { type: 'message_delta', data: { text: 'from-' } }
    yield { type: 'message_delta', data: { text: 'stream' } }
  }
}

const req: ModelRequest = { model: 'm', messages: [] }

test('no onEvent → complete()', async () => {
  const gw = new SpyGateway()
  const port = new DefaultIOPort(gw)
  const resp = await port.invokeLLM(req)
  expect(gw.completeCalled).toBe(true)
  expect(gw.streamCalled).toBe(false)
  expect(resp.content).toEqual([{ type: 'text', text: 'from-complete' }])
})

test('with onEvent → stream() + aggregate + forward events', async () => {
  const gw = new SpyGateway()
  const port = new DefaultIOPort(gw)
  const seen: ModelEvent[] = []
  const resp = await port.invokeLLM(req, (e) => seen.push(e))
  expect(gw.streamCalled).toBe(true)
  expect(gw.completeCalled).toBe(false)
  expect(resp.content).toEqual([{ type: 'text', text: 'from-stream' }])
  expect(seen).toHaveLength(2)
})
```

- [ ] **Step 2: 运行验证失败**

Run: `npx jest src/__tests__/DefaultIOPort.routing.test.ts`
Expected: FAIL（`invokeLLM` 不接第二参 / 不走 stream）

- [ ] **Step 3: 改 IIOPort 接口 + DefaultIOPort 实现**

在 `src/runtime/IOPort.ts`：

顶部加 import：
```typescript
import { v4 as uuidv4 } from 'uuid'
import type { ModelRequest, ModelResponse, ModelEvent, IModelGateway } from '../types/model.js'
import { aggregateStream } from '../gateway/StreamAggregator.js'
```

`IIOPort.invokeLLM` 签名改为（行 30）：
```typescript
  /** Invoke a language model. When `onEvent` is provided, the implementation
   *  may stream and forward token-level ModelEvents while still returning the
   *  aggregated ModelResponse. */
  invokeLLM(request: ModelRequest, onEvent?: (e: ModelEvent) => void): Promise<ModelResponse>
```

`DefaultIOPort.invokeLLM`（行 61-63）替换为：
```typescript
  invokeLLM(request: ModelRequest, onEvent?: (e: ModelEvent) => void): Promise<ModelResponse> {
    if (onEvent) {
      return aggregateStream(this.gateway.stream(request), onEvent)
    }
    return this.gateway.complete(request)
  }
```

- [ ] **Step 4: 运行验证通过**

Run: `npx jest src/__tests__/DefaultIOPort.routing.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/runtime/IOPort.ts src/__tests__/DefaultIOPort.routing.test.ts
git commit -m "feat(stream): IIOPort.invokeLLM 加 onEvent; DefaultIOPort 按需选路 (#80)"
```

---

## Task 6: RecordingIOPort / ReplayingIOPort 透传 onEvent（录制/回放不变）

**Files:**
- Modify: `src/trace/RecordingIOPort.ts:150`
- Modify: `src/trace/ReplayingIOPort.ts`（invokeLLM 签名）
- Test: `src/__tests__/RecordingIOPort.stream.test.ts`

- [ ] **Step 1: 写失败测试（Recording 透传 onEvent，且仍录完整 llm.responded）**

新建 `src/__tests__/RecordingIOPort.stream.test.ts`：

```typescript
import { RecordingIOPort } from '../trace/RecordingIOPort'
import { DefaultIOPort } from '../runtime/IOPort'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'

class StreamGateway implements IModelGateway {
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    return { content: [{ type: 'text', text: 'X' }], toolCalls: [] }
  }
  async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
    yield { type: 'message_delta', data: { text: 'he' } }
    yield { type: 'message_delta', data: { text: 'llo' } }
  }
}

test('Recording forwards onEvent and still records aggregated llm.responded', async () => {
  const store = new MemoryEventStore()
  const inner = new DefaultIOPort(new StreamGateway())
  const rec = new RecordingIOPort(inner, store, 'run1')
  const seen: ModelEvent[] = []
  const resp = await rec.invokeLLM({ model: 'm', messages: [] }, (e) => seen.push(e))

  expect(seen).toHaveLength(2)                 // deltas forwarded for UI
  expect(resp.content).toEqual([{ type: 'text', text: 'hello' }])  // aggregated

  const events = await store.readByRunId('run1')
  const responded = events.find(e => e.type === 'llm.responded')
  expect(responded).toBeDefined()             // full response still recorded
  expect((responded!.payload as { response: ModelResponse }).response.content)
    .toEqual([{ type: 'text', text: 'hello' }])
})
```

- [ ] **Step 2: 运行验证失败**

Run: `npx jest src/__tests__/RecordingIOPort.stream.test.ts`
Expected: FAIL（`invokeLLM` 不接 onEvent，deltas 未透传）

- [ ] **Step 3: 改 RecordingIOPort.invokeLLM 透传**

`src/trace/RecordingIOPort.ts` 顶部 import 加：
```typescript
import type { ModelRequest, ModelResponse, ModelEvent } from '../types/model.js'
```
（若已 import ModelRequest/ModelResponse，仅补 `ModelEvent`）

`invokeLLM` 签名（行 150）改为：
```typescript
  async invokeLLM(request: ModelRequest, onEvent?: (e: ModelEvent) => void): Promise<ModelResponse> {
```
并把内部调用（行 166）改为：
```typescript
    const response = await this.inner.invokeLLM(request, onEvent)
```
其余录制逻辑完全不动（仍录完整 response）。

- [ ] **Step 4: 改 ReplayingIOPort.invokeLLM 签名（忽略 onEvent）**

`src/trace/ReplayingIOPort.ts` 的 `invokeLLM` 签名加 `_onEvent`（replay 不流式）：
```typescript
  async invokeLLM(request: ModelRequest, _onEvent?: (e: import('../types/model.js').ModelEvent) => void): Promise<ModelResponse> {
```
方法体不变（仍从 cache 出队）。

- [ ] **Step 5: 运行验证通过 + replay 回归**

Run: `npx jest src/__tests__/RecordingIOPort.stream.test.ts`
Expected: PASS

Run: `npm run test:e2e:deterministic`
Expected: PASS（determinism 不受影响——这是 replay 零改动的硬证据）

- [ ] **Step 6: Commit**

```bash
git add src/trace/RecordingIOPort.ts src/trace/ReplayingIOPort.ts src/__tests__/RecordingIOPort.stream.test.ts
git commit -m "feat(stream): Recording 透传 onEvent, Replaying 忽略; replay 不变 (#80)"
```

---

## Task 7: AgentRuntime 透传 onModelEvent 到 invokeLLM

**Files:**
- Modify: `src/runtime/AgentRuntime.ts`（构造接收 onModelEvent；行 962 调用处）
- Test: `src/__tests__/AgentRuntime.stream.test.ts`

- [ ] **Step 1: 写失败测试（runtime 跑一轮，onModelEvent 收到 deltas）**

新建 `src/__tests__/AgentRuntime.stream.test.ts`，参考现有 `AgentRuntime.test.ts` 的 SequentialGateway 模式，但 gateway 用 stream：

```typescript
import { AgentRuntime } from '../runtime/AgentRuntime'
import { DefaultIOPort } from '../runtime/IOPort'
import { MemoryStore } from '../store/MemoryStore'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'
import type { AgentConfig } from '../types/agent'

class TextStreamGateway implements IModelGateway {
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    return { content: [{ type: 'text', text: 'hello world' }], toolCalls: [] }
  }
  async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
    yield { type: 'message_delta', data: { text: 'hello ' } }
    yield { type: 'message_delta', data: { text: 'world' } }
  }
}

const config: AgentConfig = {
  agentId: 'a', version: '0.0.1',
  systemPrompt: 'sys',
  fsm: { states: [{ name: 'respond', type: 'llm', instructions: 'go' }] } as AgentConfig['fsm'],
  model: { provider: 'volcengine', model: 'm', adapter: 'openai-compatible' },
}

test('onModelEvent receives token deltas during a run', async () => {
  const seen: ModelEvent[] = []
  const runtime = new AgentRuntime({
    config,
    goal: 'hi', input: 'hi',
    stateStore: new MemoryStore(),
    recorder: new InMemoryRecorder(undefined, 'a'),
    ioPort: new DefaultIOPort(new TextStreamGateway()),
    onModelEvent: (e) => seen.push(e),
  })
  const result = await runtime.run('hi')
  expect(result.status).toBe('completed')
  expect(seen.filter(e => e.type === 'message_delta')).toHaveLength(2)
})
```

> 注：`AgentRuntimeOptions` 的确切字段以 `src/runtime/AgentRuntime.ts` 构造为准（参考现有 AgentRuntime.test.ts 的 new AgentRuntime({...}) 用法对齐字段名）。若 fsm/config 字段名不同，按实际类型调整。

- [ ] **Step 2: 运行验证失败**

Run: `npx jest src/__tests__/AgentRuntime.stream.test.ts`
Expected: FAIL（onModelEvent 不是构造选项 / deltas 未收到）

- [ ] **Step 3: AgentRuntime 接收并透传**

在 `src/runtime/AgentRuntime.ts`：
- `AgentRuntimeOptions`（构造选项 interface）加：
  ```typescript
  onModelEvent?: (e: import('../types/model.js').ModelEvent) => void
  ```
- 构造函数内（`this.ioPort = opts.ioPort` 附近，约行 137）加：
  ```typescript
  this.onModelEvent = opts.onModelEvent
  ```
- 类字段声明（private 区，约行 99 附近）加：
  ```typescript
  private readonly onModelEvent?: (e: import('../types/model.js').ModelEvent) => void
  ```
- 调用处（行 962）改为：
  ```typescript
  const response = await this.ioPort.invokeLLM(request, this.onModelEvent)
  ```

- [ ] **Step 4: 运行验证通过**

Run: `npx jest src/__tests__/AgentRuntime.stream.test.ts`
Expected: PASS

- [ ] **Step 5: 跑 AgentRuntime 原测试确保未回归**

Run: `npx jest src/__tests__/AgentRuntime.test.ts`
Expected: PASS（无 onModelEvent 时仍走 complete()，行为不变）

- [ ] **Step 6: Commit**

```bash
git add src/runtime/AgentRuntime.ts src/__tests__/AgentRuntime.stream.test.ts
git commit -m "feat(stream): AgentRuntime 透传 onModelEvent 到 invokeLLM (#80)"
```

---

## Task 8: Milkie.invoke 接 onModelEvent，沿链传到 AgentRuntime

**Files:**
- Modify: `src/types/common.ts:20`（AgentInvokeRequest）
- Modify: `src/runtime/Milkie.ts`（invoke 把 onModelEvent 传入 AgentRuntime）
- Test: `src/__tests__/Milkie.stream.test.ts`

- [ ] **Step 1: 写失败测试（端到端：invoke 带 onModelEvent 收到 deltas）**

新建 `src/__tests__/Milkie.stream.test.ts`：

```typescript
import { Milkie } from '../runtime/Milkie'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'
import type { AgentConfig } from '../types/agent'

class TextStreamGateway implements IModelGateway {
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    return { content: [{ type: 'text', text: 'hi there' }], toolCalls: [] }
  }
  async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
    yield { type: 'message_delta', data: { text: 'hi ' } }
    yield { type: 'message_delta', data: { text: 'there' } }
  }
}

const config: AgentConfig = {
  agentId: 'poc', version: '0.0.1', systemPrompt: 'sys',
  fsm: { states: [{ name: 'respond', type: 'llm', instructions: 'go' }] } as AgentConfig['fsm'],
  model: { provider: 'volcengine', model: 'm', adapter: 'openai-compatible' },
}

test('invoke with onModelEvent streams deltas and still returns aggregated output', async () => {
  const milkie = new Milkie({ gateway: new TextStreamGateway() })
  milkie.registerAgent(config)
  const seen: ModelEvent[] = []
  const result = await milkie.invoke({ agentId: 'poc', goal: 'hi', input: 'hi', onModelEvent: (e) => seen.push(e) })
  expect(result.status).toBe('completed')
  expect(result.output).toContain('there')
  expect(seen.filter(e => e.type === 'message_delta')).toHaveLength(2)
})

test('invoke without onModelEvent uses complete() (no streaming)', async () => {
  const milkie = new Milkie({ gateway: new TextStreamGateway() })
  milkie.registerAgent(config)
  const result = await milkie.invoke({ agentId: 'poc', goal: 'hi', input: 'hi' })
  expect(result.status).toBe('completed')
})
```

- [ ] **Step 2: 运行验证失败**

Run: `npx jest src/__tests__/Milkie.stream.test.ts`
Expected: FAIL（`onModelEvent` 不是 AgentInvokeRequest 字段 / 未传入 runtime）

- [ ] **Step 3: AgentInvokeRequest 加字段**

`src/types/common.ts` 的 `AgentInvokeRequest`（行 20-25）加：
```typescript
export interface AgentInvokeRequest {
  agentId: string
  goal: string
  input: string
  contextId?: string
  /** When provided, the run streams token-level ModelEvents to this callback. */
  onModelEvent?: (e: ModelEvent) => void
}
```
并在文件顶部 import：
```typescript
import type { ModelEvent } from './model.js'
```

- [ ] **Step 4: Milkie.invoke 把 onModelEvent 传入 AgentRuntime**

`src/runtime/Milkie.ts` 有 3 个 `new AgentRuntime` 构造点（行 226 invoke / 行 311 resume / 行 434 replay）。**只改 `invoke()` 的那个（行 226）** —— resume/replay 不接 onModelEvent（resume 可作 follow-up，replay 永不流式）。在行 226 的选项对象加一行：
```typescript
      ...(request.onModelEvent ? { onModelEvent: request.onModelEvent } : {}),
```

- [ ] **Step 5: 运行验证通过**

Run: `npx jest src/__tests__/Milkie.stream.test.ts`
Expected: PASS（两个用例）

- [ ] **Step 6: Commit**

```bash
git add src/types/common.ts src/runtime/Milkie.ts src/__tests__/Milkie.stream.test.ts
git commit -m "feat(stream): Milkie.invoke 接 onModelEvent 沿链透传 (#80)"
```

---

## Task 9: 全量回归 + 选路等价性 + determinism

**Files:** 无新增，只跑测试

- [ ] **Step 1: 全量单测**

Run: `npm run test:unit`
Expected: 全绿（含原 38 基线 + 新增）

- [ ] **Step 2: determinism replay 回归（replay 零改动的硬证据）**

Run: `npm run test:e2e:deterministic`
Expected: PASS

- [ ] **Step 3: build 通过（类型检查）**

Run: `npm run build`
Expected: 0 errors

- [ ] **Step 4: 选路等价性手测（聚合 == complete）**

新建临时断言（或加进 Milkie.stream.test.ts）：同一 TextStreamGateway，complete() 的 content 与 stream 聚合的 content 相等。已在 Task 8 用例隐含覆盖（output 一致）。确认即可。

- [ ] **Step 5: 若全绿，最终 commit**

```bash
git add -A && git commit -m "test(stream): 全量回归 + determinism 绿 (#80)"
```

---

## Task 10: 更新 PoC 验证流式可用（可选，端到端真实证据）

**Files:** 无（用 alfred 侧 /tmp/milkie-poc 的 sidecar 改造验证，不进 milkie 仓库）

- [ ] **Step 1: 改 sidecar `/chat` 传 onModelEvent**

在 sidecar 的 `milkie.invoke({...})` 加 `onModelEvent: (e) => broadcast(e)`，把 ModelEvent 经 SSE 推出去。

- [ ] **Step 2: 跑 PoC，断言 SSE 收到 message_delta**

Expected: 之前 PoC 只有 llm.responded；现在应能收到多个 `message_delta`（逐 token）。这是 #80 价值的端到端实证。

> 注：此 Task 在 alfred 侧 /tmp 隔离环境做，不提交进 milkie 仓库；仅作交付验证。

---

## Self-Review 检查

- **Spec 覆盖**：设计 §4(a) 选路信号→Task 8；(b) StreamAggregator→Task 1-2；(c) 两 adapter stream()→Task 3-4；(d) IOPort 选路→Task 5；RecordingIOPort 透传→Task 6；AgentRuntime 链→Task 7；replay 安全→Task 6 Step 5 + Task 9 Step 2。✅ 全覆盖。
- **占位符**：无 TBD/TODO；所有代码步骤含完整代码。Task 7/8 对 AgentRuntimeOptions/Milkie.invoke 的字段名标注"以实际类型为准"——因这两处确切结构需实现时对照，但给了精确行号与改法。
- **类型一致**：`onModelEvent`/`onEvent`/`aggregateStream`/`ModelEvent` 联合类型字段（message_delta/tool_call_start/tool_call_delta/tool_call_done/usage）在 Task 1-8 命名一致；与 `src/types/model.ts:32-38` 既有定义对齐（不改类型，决策 2）。
- **风险点**：Task 3 OpenAI `stream_options` 与 SDK 类型可能需 `as` 断言（已用）；Task 4 Anthropic parseStreamEvent 跨事件状态用实例字段（已说明清理时机）；Task 7/8 构造字段名需对照实际——这三处是实现时最易卡点，已标注。
