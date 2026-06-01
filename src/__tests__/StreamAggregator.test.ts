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

// ===========================================================================
// tool_call 聚合完备测试
// ===========================================================================

// ---------------------------------------------------------------------------
// TC-1. 正常单 tool_call 全流程（delta 拼接 + done 无 input）
// ---------------------------------------------------------------------------
describe('StreamAggregator — tool_call TC-1: 正常单 tool_call 全流程', () => {
  test('start → delta(片段1) → delta(片段2) → done(无 input)，toolCalls 和 content 正确', async () => {
    const input: ModelEvent[] = [
      { type: 'tool_call_start', data: { toolCallId: 'tc-1', name: 'search' } },
      { type: 'tool_call_delta', data: { toolCallId: 'tc-1', delta: '{"k"' } },
      { type: 'tool_call_delta', data: { toolCallId: 'tc-1', delta: ':"v"}' } },
      { type: 'tool_call_done',  data: { toolCallId: 'tc-1', input: undefined } },
    ]
    const received: ModelEvent[] = []
    const result = await aggregateStream(events(input), (e) => received.push(e))

    // onEvent 收到全部 4 个事件
    expect(received).toHaveLength(4)
    expect(received).toEqual(input)

    // toolCalls 正确
    expect(result.toolCalls).toHaveLength(1)
    expect(result.toolCalls[0]).toEqual({ id: 'tc-1', name: 'search', input: { k: 'v' } })

    // content 含对应 tool_use
    expect(result.content).toHaveLength(1)
    expect(result.content[0]).toEqual({
      type: 'tool_use',
      id: 'tc-1',
      name: 'search',
      input: { k: 'v' },
    })
  })
})

// ---------------------------------------------------------------------------
// TC-2. done 携带 input 时以其为权威，覆盖 delta 拼出的 argsBuf
// ---------------------------------------------------------------------------
describe('StreamAggregator — tool_call TC-2: done.input authoritative', () => {
  test('delta 累加错误 JSON，done 携带 input:{correct:true}，最终 input 以 done 为准', async () => {
    const input: ModelEvent[] = [
      { type: 'tool_call_start', data: { toolCallId: 'tc-2', name: 'calc' } },
      { type: 'tool_call_delta', data: { toolCallId: 'tc-2', delta: '{"wrong":"data"}' } },
      { type: 'tool_call_done',  data: { toolCallId: 'tc-2', input: { correct: true } } },
    ]
    const result = await aggregateStream(events(input))

    expect(result.toolCalls).toHaveLength(1)
    const tc2 = result.toolCalls[0]!
    expect(tc2.input).toEqual({ correct: true })

    expect(result.content[0]).toEqual({
      type: 'tool_use',
      id: 'tc-2',
      name: 'calc',
      input: { correct: true },
    })
  })
})

// ---------------------------------------------------------------------------
// TC-3. JSON.parse 失败 → input 回退到 {}
// ---------------------------------------------------------------------------
describe('StreamAggregator — tool_call TC-3: JSON.parse 失败回退 {}', () => {
  test('delta 为非法 JSON，done 无 input，toolCalls[0].input = {}', async () => {
    const input: ModelEvent[] = [
      { type: 'tool_call_start', data: { toolCallId: 'tc-3', name: 'broken' } },
      { type: 'tool_call_delta', data: { toolCallId: 'tc-3', delta: 'invalid-json{' } },
      { type: 'tool_call_done',  data: { toolCallId: 'tc-3', input: undefined } },
    ]
    const received: ModelEvent[] = []
    const result = await aggregateStream(events(input), (e) => received.push(e))

    // onEvent 透传验证
    expect(received).toHaveLength(3)

    expect(result.toolCalls).toHaveLength(1)
    const tc3 = result.toolCalls[0]!
    expect(tc3.input).toEqual({})

    // content 里也是 {}
    expect(result.content[0]).toEqual({
      type: 'tool_use',
      id: 'tc-3',
      name: 'broken',
      input: {},
    })
  })
})

// ---------------------------------------------------------------------------
// TC-4. delta 为非 string 类型 → JSON.stringify 后聚合
// ---------------------------------------------------------------------------
describe('StreamAggregator — tool_call TC-4: delta 非 string 类型', () => {
  test('tool_call_delta.data.delta 为对象时，JSON.stringify 后追加到 argsBuf，最终 input 等于该对象', async () => {
    // delta 是对象 {key:'val'}，JSON.stringify 得 '{"key":"val"}'，parse 后还原
    const input: ModelEvent[] = [
      { type: 'tool_call_start', data: { toolCallId: 'tc-4', name: 'obj-delta' } },
      { type: 'tool_call_delta', data: { toolCallId: 'tc-4', delta: { key: 'val' } } },
      { type: 'tool_call_done',  data: { toolCallId: 'tc-4', input: undefined } },
    ]
    const result = await aggregateStream(events(input))

    expect(result.toolCalls).toHaveLength(1)
    const tc4 = result.toolCalls[0]!
    // argsBuf = '{"key":"val"}' → JSON.parse → {key:'val'}
    expect(tc4.input).toEqual({ key: 'val' })

    expect(result.content[0]).toEqual({
      type: 'tool_use',
      id: 'tc-4',
      name: 'obj-delta',
      input: { key: 'val' },
    })
  })
})

// ---------------------------------------------------------------------------
// TC-5. 同 id 重复 tool_call_start 被忽略，保留首个 name
// ---------------------------------------------------------------------------
describe('StreamAggregator — tool_call TC-5: 重复 start 被忽略', () => {
  test('同 toolCallId 发两次 start，toolCalls.length === 1，保留首个 name', async () => {
    const input: ModelEvent[] = [
      { type: 'tool_call_start', data: { toolCallId: 'tc-5', name: 'first-name' } },
      { type: 'tool_call_start', data: { toolCallId: 'tc-5', name: 'second-name' } },
      { type: 'tool_call_delta', data: { toolCallId: 'tc-5', delta: '{"x":1}' } },
      { type: 'tool_call_done',  data: { toolCallId: 'tc-5', input: undefined } },
    ]
    const result = await aggregateStream(events(input))

    expect(result.toolCalls).toHaveLength(1)
    const tc5 = result.toolCalls[0]!
    expect(tc5.name).toBe('first-name')
    expect(tc5.input).toEqual({ x: 1 })
  })
})

// ---------------------------------------------------------------------------
// TC-6. 并行多 tool_call 保序且 buffer 隔离
// ---------------------------------------------------------------------------
describe('StreamAggregator — tool_call TC-6: 并行多 tool_call 保序', () => {
  test('start(a)→start(b)→delta(a)→delta(b)→done(a)→done(b)，顺序为 a 在前，buffer 隔离', async () => {
    const input: ModelEvent[] = [
      { type: 'tool_call_start', data: { toolCallId: 'a', name: 'tool-a' } },
      { type: 'tool_call_start', data: { toolCallId: 'b', name: 'tool-b' } },
      { type: 'tool_call_delta', data: { toolCallId: 'a', delta: '{"f":"x"}' } },
      { type: 'tool_call_delta', data: { toolCallId: 'b', delta: '{"p":"y"}' } },
      { type: 'tool_call_done',  data: { toolCallId: 'a', input: undefined } },
      { type: 'tool_call_done',  data: { toolCallId: 'b', input: undefined } },
    ]
    const result = await aggregateStream(events(input))

    expect(result.toolCalls).toHaveLength(2)
    // 顺序：a 在前
    expect(result.toolCalls[0]).toEqual({ id: 'a', name: 'tool-a', input: { f: 'x' } })
    expect(result.toolCalls[1]).toEqual({ id: 'b', name: 'tool-b', input: { p: 'y' } })

    // content 顺序一致
    expect(result.content[0]).toEqual({ type: 'tool_use', id: 'a', name: 'tool-a', input: { f: 'x' } })
    expect(result.content[1]).toEqual({ type: 'tool_use', id: 'b', name: 'tool-b', input: { p: 'y' } })
  })
})

// ---------------------------------------------------------------------------
// TC-7. text + tool_call 共存，content 顺序：text 在前
// ---------------------------------------------------------------------------
describe('StreamAggregator — tool_call TC-7: text + tool_call 共存', () => {
  test('message_delta 在前，tool_call 在后，content = [text, tool_use]', async () => {
    const input: ModelEvent[] = [
      { type: 'message_delta',   data: { text: 'let me search' } },
      { type: 'tool_call_start', data: { toolCallId: 'tc-7', name: 'web_search' } },
      { type: 'tool_call_delta', data: { toolCallId: 'tc-7', delta: '{"q":"ts"}' } },
      { type: 'tool_call_done',  data: { toolCallId: 'tc-7', input: undefined } },
    ]
    const received: ModelEvent[] = []
    const result = await aggregateStream(events(input), (e) => received.push(e))

    // onEvent 透传验证
    expect(received).toHaveLength(4)

    expect(result.content).toHaveLength(2)
    expect(result.content[0]).toEqual({ type: 'text', text: 'let me search' })
    expect(result.content[1]).toEqual({
      type: 'tool_use',
      id: 'tc-7',
      name: 'web_search',
      input: { q: 'ts' },
    })

    expect(result.toolCalls).toHaveLength(1)
    expect(result.toolCalls[0]).toEqual({ id: 'tc-7', name: 'web_search', input: { q: 'ts' } })
  })
})

// ---------------------------------------------------------------------------
// TC-8. 空 argsBuf（start → done，无 delta，无 done.input）→ input={}
// ---------------------------------------------------------------------------
describe('StreamAggregator — tool_call TC-8: 空 argsBuf → input={}', () => {
  test('只有 start 和 done（无 delta，done 无 input），toolCalls[0].input = {}', async () => {
    const input: ModelEvent[] = [
      { type: 'tool_call_start', data: { toolCallId: 'tc-8', name: 'ping' } },
      { type: 'tool_call_done',  data: { toolCallId: 'tc-8', input: undefined } },
    ]
    const result = await aggregateStream(events(input))

    expect(result.toolCalls).toHaveLength(1)
    expect(result.toolCalls[0]).toEqual({ id: 'tc-8', name: 'ping', input: {} })

    expect(result.content[0]).toEqual({
      type: 'tool_use',
      id: 'tc-8',
      name: 'ping',
      input: {},
    })
  })
})

// ===========================================================================
// 健壮性边界：乱序 / null input
// ===========================================================================

// ---------------------------------------------------------------------------
// ROBUSTNESS-1. 孤儿 delta / done（无 start）被安全忽略
// ---------------------------------------------------------------------------
describe('StreamAggregator — robustness / out-of-order: 孤儿 delta/done 被安全忽略', () => {
  test('只发 tool_call_delta（无 start）→ 不抛错，toolCalls=[], content=[]', async () => {
    const input: ModelEvent[] = [
      { type: 'tool_call_delta', data: { toolCallId: 'x', delta: '{"orphan":true}' } },
    ]
    const result = await aggregateStream(events(input))

    // 不抛错，Map 中无对应 id，delta 被静默跳过
    expect(result.toolCalls).toEqual([])
    expect(result.content).toEqual([])
  })

  test('只发 tool_call_done（无 start）→ 不抛错，toolCalls=[], content=[]', async () => {
    const input: ModelEvent[] = [
      { type: 'tool_call_done', data: { toolCallId: 'y', input: { should: 'be ignored' } } },
    ]
    const result = await aggregateStream(events(input))

    // 不抛错，Map 中无对应 id，done 被静默跳过
    expect(result.toolCalls).toEqual([])
    expect(result.content).toEqual([])
  })
})

// ---------------------------------------------------------------------------
// ROBUSTNESS-2. done.input 为 null → null 是权威值，toolCalls[0].input === null
// ---------------------------------------------------------------------------
describe('StreamAggregator — robustness / null input: done.input=null 的语义契约', () => {
  test('start→done(input:null)，null !== undefined 走权威分支，toolCalls[0].input === null', async () => {
    const input: ModelEvent[] = [
      { type: 'tool_call_start', data: { toolCallId: 't', name: 'n' } },
      { type: 'tool_call_done',  data: { toolCallId: 't', input: null } },
    ]
    const result = await aggregateStream(events(input))

    expect(result.toolCalls).toHaveLength(1)
    // null !== undefined → JSON.stringify(null)='null' → JSON.parse('null')=null
    expect(result.toolCalls[0]!.input).toBeNull()
    expect(result.content[0]).toEqual({
      type: 'tool_use',
      id: 't',
      name: 'n',
      input: null,
    })
  })
})

// ---------------------------------------------------------------------------
// ROBUSTNESS-3. 先 delta 后 start（乱序）→ delta 被忽略，最终 input={}
// ---------------------------------------------------------------------------
describe('StreamAggregator — robustness / out-of-order: delta 先于 start 到达', () => {
  test('delta(z) 在 start(z) 之前：delta 静默丢弃，argsBuf 为空，最终 input={}', async () => {
    const input: ModelEvent[] = [
      // delta 先到，此时 Map 中还没有 'z'，守卫 if(acc) 静默跳过
      { type: 'tool_call_delta', data: { toolCallId: 'z', delta: '{"a":1}' } },
      // start 后到，建立空 argsBuf
      { type: 'tool_call_start', data: { toolCallId: 'z', name: 'late-start' } },
      { type: 'tool_call_done',  data: { toolCallId: 'z', input: undefined } },
    ]
    const result = await aggregateStream(events(input))

    expect(result.toolCalls).toHaveLength(1)
    // delta 被丢弃 → argsBuf='' → 空字符串走 falsy 分支 → input={}
    expect(result.toolCalls[0]).toEqual({ id: 'z', name: 'late-start', input: {} })
    expect(result.content[0]).toEqual({
      type: 'tool_use',
      id: 'z',
      name: 'late-start',
      input: {},
    })
  })
})
