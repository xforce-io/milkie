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
