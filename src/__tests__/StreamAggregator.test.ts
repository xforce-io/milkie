import { aggregateStream } from '../gateway/StreamAggregator'
import type { ModelEvent } from '../types/model'

async function* events(list: ModelEvent[]): AsyncIterable<ModelEvent> {
  for (const e of list) yield e
}

// ---------------------------------------------------------------------------
// 1. 多个 message_delta 聚合 + usage 正确 + onEvent 转发每个事件
// ---------------------------------------------------------------------------
describe('StreamAggregator — message_delta aggregation', () => {
  test('多个 message_delta 聚合成一个 text content，usage 正确，onEvent 收到全部事件', async () => {
    const input: ModelEvent[] = [
      { type: 'message_delta', data: { text: 'Hello' } },
      { type: 'message_delta', data: { text: ', ' } },
      { type: 'message_delta', data: { text: 'world!' } },
      { type: 'usage', data: { inputTokens: 10, outputTokens: 5 } },
    ]
    const received: ModelEvent[] = []
    const result = await aggregateStream(events(input), (e: ModelEvent) => received.push(e))

    // onEvent 必须收到全部事件，顺序一致
    expect(received).toHaveLength(input.length)
    expect(received).toEqual(input)

    // text 聚合正确
    expect(result.content).toHaveLength(1)
    expect(result.content[0]).toEqual({ type: 'text', text: 'Hello, world!' })

    // toolCalls 为空
    expect(result.toolCalls).toEqual([])

    // usage 正确
    expect(result.usage).toEqual({ inputTokens: 10, outputTokens: 5 })

    // finishReason 未设置
    expect(result.finishReason).toBeUndefined()
  })
})

// ---------------------------------------------------------------------------
// 2. onEvent 可选（不传时正常聚合，不抛错）
// ---------------------------------------------------------------------------
describe('StreamAggregator — onEvent optional', () => {
  test('没有 onEvent 时也能正常聚合', async () => {
    const input: ModelEvent[] = [
      { type: 'message_delta', data: { text: 'Hello' } },
      { type: 'usage', data: { inputTokens: 3, outputTokens: 2 } },
    ]
    const result = await aggregateStream(events(input))
    expect(result.content).toEqual([{ type: 'text', text: 'Hello' }])
    expect(result.usage).toEqual({ inputTokens: 3, outputTokens: 2 })
  })
})

// ---------------------------------------------------------------------------
// 3. 空流
// ---------------------------------------------------------------------------
describe('StreamAggregator — empty stream', () => {
  test('空流 → content=[], toolCalls=[], usage undefined, finishReason undefined', async () => {
    const result = await aggregateStream(events([]))
    expect(result.content).toEqual([])
    expect(result.toolCalls).toEqual([])
    expect(result.usage).toBeUndefined()
    expect(result.finishReason).toBeUndefined()
  })
})

// ---------------------------------------------------------------------------
// 4. 只有 usage 没有文本
// ---------------------------------------------------------------------------
describe('StreamAggregator — usage only', () => {
  test('只有 usage 事件 → content=[], usage 正确', async () => {
    const input: ModelEvent[] = [
      { type: 'usage', data: { inputTokens: 100, outputTokens: 0, cost: 0.001 } },
    ]
    const result = await aggregateStream(events(input))
    expect(result.content).toEqual([])
    expect(result.toolCalls).toEqual([])
    expect(result.usage).toEqual({ inputTokens: 100, outputTokens: 0, cost: 0.001 })
  })
})

// ---------------------------------------------------------------------------
// 5. error 事件 → finishReason='error'，已聚合内容仍返回
// ---------------------------------------------------------------------------
describe('StreamAggregator — error event', () => {
  test('error 事件设置 finishReason=error，已聚合文本仍在', async () => {
    const input: ModelEvent[] = [
      { type: 'message_delta', data: { text: 'partial' } },
      { type: 'error', data: { code: 'TIMEOUT', message: 'timed out', retryable: true } },
    ]
    const received: ModelEvent[] = []
    const result = await aggregateStream(events(input), (e: ModelEvent) => received.push(e))

    expect(received).toHaveLength(2)
    expect(result.finishReason).toBe('error')
    expect(result.content).toEqual([{ type: 'text', text: 'partial' }])
  })

  test('多个 error 事件，finishReason 保持 error（只设一次，不重复覆盖已设值）', async () => {
    const input: ModelEvent[] = [
      { type: 'error', data: { code: 'E1', message: 'first' } },
      { type: 'error', data: { code: 'E2', message: 'second' } },
    ]
    const result = await aggregateStream(events(input))
    expect(result.finishReason).toBe('error')
  })
})

// ---------------------------------------------------------------------------
// 6. 边界：message_delta 的 text 为空字符串
//    语义：单个空 delta 不产生 text content（累加后总文本为空则跳过）
// ---------------------------------------------------------------------------
describe('StreamAggregator — empty text delta boundary', () => {
  test('所有 message_delta 的 text 都是空字符串 → 不产生 text content', async () => {
    const input: ModelEvent[] = [
      { type: 'message_delta', data: { text: '' } },
      { type: 'message_delta', data: { text: '' } },
    ]
    const result = await aggregateStream(events(input))
    expect(result.content).toEqual([])
    expect(result.toolCalls).toEqual([])
  })

  test('混合空和非空 delta → 只聚合非空部分，仍产生 text content', async () => {
    const input: ModelEvent[] = [
      { type: 'message_delta', data: { text: '' } },
      { type: 'message_delta', data: { text: 'hi' } },
      { type: 'message_delta', data: { text: '' } },
    ]
    const result = await aggregateStream(events(input))
    expect(result.content).toEqual([{ type: 'text', text: 'hi' }])
  })
})

// ---------------------------------------------------------------------------
// 7. usage 含可选字段（cache tokens）
// ---------------------------------------------------------------------------
describe('StreamAggregator — usage with cache fields', () => {
  test('usage 含 cacheReadTokens / cacheCreationTokens 正确保留', async () => {
    const input: ModelEvent[] = [
      {
        type: 'usage',
        data: {
          inputTokens: 200,
          outputTokens: 50,
          cacheReadTokens: 30,
          cacheCreationTokens: 10,
        },
      },
    ]
    const result = await aggregateStream(events(input))
    expect(result.usage).toEqual({
      inputTokens: 200,
      outputTokens: 50,
      cacheReadTokens: 30,
      cacheCreationTokens: 10,
    })
  })
})

// ---------------------------------------------------------------------------
// 8. onEvent 转发顺序严格一致（多种事件类型混合）
// ---------------------------------------------------------------------------
describe('StreamAggregator — onEvent ordering', () => {
  test('混合事件类型时 onEvent 收到顺序与输入完全一致', async () => {
    const input: ModelEvent[] = [
      { type: 'message_delta', data: { text: 'A' } },
      { type: 'usage', data: { inputTokens: 1, outputTokens: 1 } },
      { type: 'message_delta', data: { text: 'B' } },
      { type: 'error', data: { code: 'X', message: 'oops' } },
    ]
    const received: ModelEvent[] = []
    await aggregateStream(events(input), (e: ModelEvent) => received.push(e))
    expect(received).toEqual(input)
  })
})
