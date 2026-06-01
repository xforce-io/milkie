import { AnthropicAdapter } from '../gateway/AnthropicAdapter'
import type { ModelEvent } from '../types/model'

// parseStreamEvent is a private generator. Cast and feed it raw events directly.
// NOTE: cross-event state (partial_json buffering) lives on the instance, so a
// fresh adapter is used per stateful test (or events are fed in order).
function parse(adapter: AnthropicAdapter, raw: unknown): ModelEvent[] {
  return [...(adapter as unknown as { parseStreamEvent(e: unknown): Iterable<ModelEvent> }).parseStreamEvent(raw)]
}

function freshAdapter(): AnthropicAdapter {
  return new AnthropicAdapter({ apiKey: 'sk-test' })
}

describe('AnthropicAdapter.parseStreamEvent — text delta', () => {
  test('1. text_delta → message_delta', () => {
    const events = parse(freshAdapter(), {
      type:  'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: 'hello' },
    })
    expect(events).toEqual([{ type: 'message_delta', data: { text: 'hello' } }])
  })

  test('text_delta with empty text → no event', () => {
    const events = parse(freshAdapter(), {
      type:  'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: '' },
    })
    expect(events).toEqual([])
  })
})

describe('AnthropicAdapter.parseStreamEvent — tool_use lifecycle', () => {
  test('2. tool_use content_block_start → tool_call_start (id/name correct)', () => {
    const events = parse(freshAdapter(), {
      type:          'content_block_start',
      index:         0,
      content_block: { type: 'tool_use', id: 'toolu_abc', name: 'get_weather' },
    })
    expect(events).toEqual([
      { type: 'tool_call_start', data: { toolCallId: 'toolu_abc', name: 'get_weather' } },
    ])
  })

  test('3. input_json_delta after start → tool_call_delta (delta=partial_json, toolCallId correct)', () => {
    const adapter = freshAdapter()
    parse(adapter, {
      type:          'content_block_start',
      index:         0,
      content_block: { type: 'tool_use', id: 'toolu_1', name: 'fn' },
    })
    const events = parse(adapter, {
      type:  'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json: '{"q":' },
    })
    expect(events).toEqual([
      { type: 'tool_call_delta', data: { toolCallId: 'toolu_1', delta: '{"q":' } },
    ])
  })

  test('4. content_block_stop after start+delta → tool_call_done (input parsed)', () => {
    const adapter = freshAdapter()
    parse(adapter, {
      type:          'content_block_start',
      index:         0,
      content_block: { type: 'tool_use', id: 'toolu_2', name: 'fn' },
    })
    parse(adapter, {
      type:  'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json: '{"a":1}' },
    })
    const events = parse(adapter, { type: 'content_block_stop', index: 0 })
    expect(events).toEqual([
      { type: 'tool_call_done', data: { toolCallId: 'toolu_2', input: { a: 1 } } },
    ])
  })

  test('5. full single tool_use sequence: start → delta → delta → stop', () => {
    const adapter = freshAdapter()
    const all: ModelEvent[] = []
    all.push(...parse(adapter, {
      type:          'content_block_start',
      index:         0,
      content_block: { type: 'tool_use', id: 'toolu_seq', name: 'search' },
    }))
    all.push(...parse(adapter, {
      type:  'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json: '{"q":' },
    }))
    all.push(...parse(adapter, {
      type:  'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json: '"x"}' },
    }))
    all.push(...parse(adapter, { type: 'content_block_stop', index: 0 }))

    expect(all).toEqual([
      { type: 'tool_call_start', data: { toolCallId: 'toolu_seq', name: 'search' } },
      { type: 'tool_call_delta', data: { toolCallId: 'toolu_seq', delta: '{"q":' } },
      { type: 'tool_call_delta', data: { toolCallId: 'toolu_seq', delta: '"x"}' } },
      { type: 'tool_call_done', data: { toolCallId: 'toolu_seq', input: { q: 'x' } } },
    ])
  })

  test('6. parallel two tool_use (different index) — interleaved, isolated buffers', () => {
    const adapter = freshAdapter()
    const all: ModelEvent[] = []
    // start block 0
    all.push(...parse(adapter, {
      type:          'content_block_start',
      index:         0,
      content_block: { type: 'tool_use', id: 'toolu_A', name: 'fnA' },
    }))
    // start block 1
    all.push(...parse(adapter, {
      type:          'content_block_start',
      index:         1,
      content_block: { type: 'tool_use', id: 'toolu_B', name: 'fnB' },
    }))
    // interleaved deltas
    all.push(...parse(adapter, {
      type:  'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json: '{"a":' },
    }))
    all.push(...parse(adapter, {
      type:  'content_block_delta',
      index: 1,
      delta: { type: 'input_json_delta', partial_json: '{"b":' },
    }))
    all.push(...parse(adapter, {
      type:  'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json: '1}' },
    }))
    all.push(...parse(adapter, {
      type:  'content_block_delta',
      index: 1,
      delta: { type: 'input_json_delta', partial_json: '2}' },
    }))
    // stop in reverse order
    all.push(...parse(adapter, { type: 'content_block_stop', index: 1 }))
    all.push(...parse(adapter, { type: 'content_block_stop', index: 0 }))

    expect(all).toEqual([
      { type: 'tool_call_start', data: { toolCallId: 'toolu_A', name: 'fnA' } },
      { type: 'tool_call_start', data: { toolCallId: 'toolu_B', name: 'fnB' } },
      { type: 'tool_call_delta', data: { toolCallId: 'toolu_A', delta: '{"a":' } },
      { type: 'tool_call_delta', data: { toolCallId: 'toolu_B', delta: '{"b":' } },
      { type: 'tool_call_delta', data: { toolCallId: 'toolu_A', delta: '1}' } },
      { type: 'tool_call_delta', data: { toolCallId: 'toolu_B', delta: '2}' } },
      { type: 'tool_call_done', data: { toolCallId: 'toolu_B', input: { b: 2 } } },
      { type: 'tool_call_done', data: { toolCallId: 'toolu_A', input: { a: 1 } } },
    ])
  })
})

describe('AnthropicAdapter.parseStreamEvent — usage', () => {
  test('7. message_delta with usage → usage event (inputTokens:0, outputTokens correct)', () => {
    const events = parse(freshAdapter(), {
      type:  'message_delta',
      delta: { stop_reason: 'end_turn' },
      usage: { output_tokens: 42 },
    })
    expect(events).toEqual([
      { type: 'usage', data: { inputTokens: 0, outputTokens: 42 } },
    ])
  })

  test('message_delta with usage but missing output_tokens → outputTokens 0', () => {
    const events = parse(freshAdapter(), {
      type:  'message_delta',
      delta: { stop_reason: 'tool_use' },
      usage: {},
    })
    expect(events).toEqual([
      { type: 'usage', data: { inputTokens: 0, outputTokens: 0 } },
    ])
  })

  test('message_delta without usage → no event', () => {
    const events = parse(freshAdapter(), {
      type:  'message_delta',
      delta: { stop_reason: 'end_turn' },
    })
    expect(events).toEqual([])
  })
})

describe('AnthropicAdapter.parseStreamEvent — edge cases', () => {
  test('8. content_block_stop with malformed JSON buffer → done.input={}', () => {
    const adapter = freshAdapter()
    parse(adapter, {
      type:          'content_block_start',
      index:         0,
      content_block: { type: 'tool_use', id: 'toolu_bad', name: 'fn' },
    })
    parse(adapter, {
      type:  'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json: '{not json' },
    })
    const events = parse(adapter, { type: 'content_block_stop', index: 0 })
    expect(events).toEqual([
      { type: 'tool_call_done', data: { toolCallId: 'toolu_bad', input: {} } },
    ])
  })

  test('content_block_stop with empty buffer → done.input={}', () => {
    const adapter = freshAdapter()
    parse(adapter, {
      type:          'content_block_start',
      index:         0,
      content_block: { type: 'tool_use', id: 'toolu_empty', name: 'fn' },
    })
    const events = parse(adapter, { type: 'content_block_stop', index: 0 })
    expect(events).toEqual([
      { type: 'tool_call_done', data: { toolCallId: 'toolu_empty', input: {} } },
    ])
  })

  test('9. content_block_stop with no matching start (orphan index) → no yield, no throw', () => {
    const adapter = freshAdapter()
    let events: ModelEvent[] = []
    expect(() => { events = parse(adapter, { type: 'content_block_stop', index: 7 }) }).not.toThrow()
    expect(events).toEqual([])
  })

  test('10. text content_block_start (type!=tool_use) → no streamTools entry, no tool event', () => {
    const adapter = freshAdapter()
    const startEvents = parse(adapter, {
      type:          'content_block_start',
      index:         0,
      content_block: { type: 'text', text: '' },
    })
    expect(startEvents).toEqual([])
    // A subsequent stop on index 0 must not produce tool_call_done (no slot created).
    const stopEvents = parse(adapter, { type: 'content_block_stop', index: 0 })
    expect(stopEvents).toEqual([])
  })

  test('11. input_json_delta with no slot for index (out of order) → silently skipped, no throw', () => {
    const adapter = freshAdapter()
    let events: ModelEvent[] = []
    expect(() => {
      events = parse(adapter, {
        type:  'content_block_delta',
        index: 3,
        delta: { type: 'input_json_delta', partial_json: '{"x":1}' },
      })
    }).not.toThrow()
    expect(events).toEqual([])
  })

  test('unknown event type → no event, no throw', () => {
    const adapter = freshAdapter()
    let events: ModelEvent[] = []
    expect(() => { events = parse(adapter, { type: 'message_start', message: {} }) }).not.toThrow()
    expect(events).toEqual([])
  })

  test('tool_use content_block_start with missing id/name → empty strings', () => {
    const events = parse(freshAdapter(), {
      type:          'content_block_start',
      index:         0,
      content_block: { type: 'tool_use' },
    })
    expect(events).toEqual([
      { type: 'tool_call_start', data: { toolCallId: '', name: '' } },
    ])
  })
})

// ---------------------------------------------------------------------------
// stream() 层级：流级别 streamTools 清理测试
// ---------------------------------------------------------------------------

function adapterWithStreamEvents(events: unknown[]): AnthropicAdapter {
  const adapter = new AnthropicAdapter({ apiKey: 'sk-test' })
  ;(adapter as unknown as { client: { messages: { stream: (...a: unknown[]) => unknown } } })
    .client = { messages: { stream: () => (async function* () { for (const e of events) yield e })() } }
  return adapter
}

async function collectStream(adapter: AnthropicAdapter): Promise<ModelEvent[]> {
  const out: ModelEvent[] = []
  for await (const e of adapter.stream({ model: 'm', messages: [] })) out.push(e)
  return out
}

describe('AnthropicAdapter.stream() — 流级别 streamTools 清理', () => {
  test('12. 中断流的残留不污染下一流：第一流无 stop 留下 OLD，第二流 start NEW 正确覆盖', async () => {
    const adapter = new AnthropicAdapter({ apiKey: 'sk-test' })

    // 第一次流：content_block_start + delta，无 content_block_stop（模拟中断）
    const firstEvents = [
      { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'OLD', name: 'a' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"old":' } },
    ]
    ;(adapter as unknown as { client: { messages: { stream: (...a: unknown[]) => unknown } } })
      .client = { messages: { stream: () => (async function* () { for (const e of firstEvents) yield e })() } }
    const first = await collectStream(adapter)

    // 第一次：应有 start + delta，无 done
    expect(first).toEqual([
      { type: 'tool_call_start', data: { toolCallId: 'OLD', name: 'a' } },
      { type: 'tool_call_delta', data: { toolCallId: 'OLD', delta: '{"old":' } },
    ])

    // 第二次流：完整序列，index 0，id='NEW'
    const secondEvents = [
      { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'NEW', name: 'b' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"x":1}' } },
      { type: 'content_block_stop', index: 0 },
    ]
    ;(adapter as unknown as { client: { messages: { stream: (...a: unknown[]) => unknown } } })
      .client = { messages: { stream: () => (async function* () { for (const e of secondEvents) yield e })() } }
    const second = await collectStream(adapter)

    // 断言第二次流使用 NEW（而非 OLD 残留）
    const startEvent = second.find(e => e.type === 'tool_call_start')
    const doneEvent  = second.find(e => e.type === 'tool_call_done')
    expect(startEvent).toEqual({ type: 'tool_call_start', data: { toolCallId: 'NEW', name: 'b' } })
    expect(doneEvent).toEqual({ type: 'tool_call_done', data: { toolCallId: 'NEW', input: { x: 1 } } })
  })

  test('13. 正常流结束后 streamTools 为空（无泄漏）', async () => {
    const events = [
      { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'toolu_z', name: 'fn' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"z":9}' } },
      { type: 'content_block_stop', index: 0 },
    ]
    const adapter = adapterWithStreamEvents(events)
    await collectStream(adapter)
    // stream() finally 块保证 streamTools 清空
    expect((adapter as unknown as { streamTools: Map<unknown, unknown> }).streamTools.size).toBe(0)
  })
})
