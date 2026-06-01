import { OpenAICompatibleAdapter } from '../gateway/OpenAICompatibleAdapter'
import type { ModelEvent } from '../types/model'

// Feed a fake async stream by swapping the adapter's private client — no network.
function adapterWithChunks(chunks: unknown[]): {
  adapter: OpenAICompatibleAdapter
  captured: { params?: unknown }
} {
  const adapter = new OpenAICompatibleAdapter({ apiKey: 'sk-test' })
  const captured: { params?: unknown } = {}
  const fakeStream = (async function* () {
    for (const c of chunks) yield c
  })()
  ;(
    adapter as unknown as {
      client: { chat: { completions: { create: (...a: unknown[]) => unknown } } }
    }
  ).client = {
    chat: {
      completions: {
        create: async (params: unknown) => {
          captured.params = params
          return fakeStream
        },
      },
    },
  }
  return { adapter, captured }
}

async function collect(adapter: OpenAICompatibleAdapter): Promise<ModelEvent[]> {
  const out: ModelEvent[] = []
  for await (const e of adapter.stream({ model: 'm', messages: [] })) out.push(e)
  return out
}

// Build a text-content chunk.
function contentChunk(text: string): unknown {
  return { choices: [{ delta: { content: text }, finish_reason: null }] }
}

// Build a tool_calls delta chunk.
function toolChunk(
  toolCalls: Array<{
    index: number
    id?: string
    name?: string
    arguments?: string
  }>,
  finish_reason: string | null = null,
): unknown {
  return {
    choices: [
      {
        delta: {
          tool_calls: toolCalls.map(tc => ({
            index:    tc.index,
            ...(tc.id !== undefined ? { id: tc.id } : {}),
            function: {
              ...(tc.name !== undefined ? { name: tc.name } : {}),
              ...(tc.arguments !== undefined ? { arguments: tc.arguments } : {}),
            },
          })),
        },
        finish_reason,
      },
    ],
  }
}

function usageChunk(usage: unknown): unknown {
  return { choices: [], usage }
}

describe('OpenAICompatibleAdapter.stream — token-level streaming', () => {
  test('request includes stream + stream_options.include_usage', async () => {
    const { adapter, captured } = adapterWithChunks([])
    await collect(adapter)
    expect(captured.params).toMatchObject({
      stream:         true,
      stream_options: { include_usage: true },
    })
  })

  test('request maps tools and tool_choice', async () => {
    const { adapter, captured } = adapterWithChunks([])
    const out: ModelEvent[] = []
    for await (const e of adapter.stream({
      model:    'm',
      messages: [],
      tools:    [{ name: 'get_weather', description: 'd', inputSchema: { type: 'object' } }],
    })) {
      out.push(e)
    }
    expect(captured.params).toMatchObject({
      tools: [
        {
          type:     'function',
          function: { name: 'get_weather', description: 'd', parameters: { type: 'object' } },
        },
      ],
      tool_choice: 'auto',
    })
  })

  test('1. pure text stream → multiple message_delta events', async () => {
    const { adapter } = adapterWithChunks([
      contentChunk('Hello'),
      contentChunk(', '),
      contentChunk('world'),
    ])
    const events = await collect(adapter)
    expect(events).toEqual([
      { type: 'message_delta', data: { text: 'Hello' } },
      { type: 'message_delta', data: { text: ', ' } },
      { type: 'message_delta', data: { text: 'world' } },
    ])
  })

  test('2. single tool_call: name on first chunk, arguments across 2 chunks, finish=tool_calls', async () => {
    const { adapter } = adapterWithChunks([
      toolChunk([{ index: 0, id: 'call_1', name: 'get_weather', arguments: '{"city":' }]),
      toolChunk([{ index: 0, arguments: '"SF"}' }]),
      toolChunk([], 'tool_calls'),
    ])
    const events = await collect(adapter)
    expect(events).toEqual([
      { type: 'tool_call_start', data: { toolCallId: 'call_1', name: 'get_weather' } },
      { type: 'tool_call_delta', data: { toolCallId: 'call_1', delta: '{"city":' } },
      { type: 'tool_call_delta', data: { toolCallId: 'call_1', delta: '"SF"}' } },
      { type: 'tool_call_done', data: { toolCallId: 'call_1', input: { city: 'SF' } } },
    ])
  })

  test('3. mixed text + tool_call', async () => {
    const { adapter } = adapterWithChunks([
      contentChunk('Let me check. '),
      toolChunk([{ index: 0, id: 'call_x', name: 'lookup', arguments: '{"q":"x"}' }]),
      toolChunk([], 'tool_calls'),
    ])
    const events = await collect(adapter)
    expect(events).toEqual([
      { type: 'message_delta', data: { text: 'Let me check. ' } },
      { type: 'tool_call_start', data: { toolCallId: 'call_x', name: 'lookup' } },
      { type: 'tool_call_delta', data: { toolCallId: 'call_x', delta: '{"q":"x"}' } },
      { type: 'tool_call_done', data: { toolCallId: 'call_x', input: { q: 'x' } } },
    ])
  })

  test('4. parallel tool_calls: two indexes with interleaved argument chunks', async () => {
    const { adapter } = adapterWithChunks([
      toolChunk([{ index: 0, id: 'call_a', name: 'tool_a', arguments: '{"a":' }]),
      toolChunk([{ index: 1, id: 'call_b', name: 'tool_b', arguments: '{"b":' }]),
      toolChunk([{ index: 0, arguments: '1}' }]),
      toolChunk([{ index: 1, arguments: '2}' }]),
      toolChunk([], 'tool_calls'),
    ])
    const events = await collect(adapter)
    // Each chunk carries one tool_call; start+delta are emitted together as the
    // chunk is processed. finish_reason then flushes done for both, in Map
    // insertion order (a before b).
    expect(events).toEqual([
      { type: 'tool_call_start', data: { toolCallId: 'call_a', name: 'tool_a' } },
      { type: 'tool_call_delta', data: { toolCallId: 'call_a', delta: '{"a":' } },
      { type: 'tool_call_start', data: { toolCallId: 'call_b', name: 'tool_b' } },
      { type: 'tool_call_delta', data: { toolCallId: 'call_b', delta: '{"b":' } },
      { type: 'tool_call_delta', data: { toolCallId: 'call_a', delta: '1}' } },
      { type: 'tool_call_delta', data: { toolCallId: 'call_b', delta: '2}' } },
      { type: 'tool_call_done', data: { toolCallId: 'call_a', input: { a: 1 } } },
      { type: 'tool_call_done', data: { toolCallId: 'call_b', input: { b: 2 } } },
    ])
  })

  test('4b. parallel tool_calls sharing one chunk → both starts in index order', async () => {
    const { adapter } = adapterWithChunks([
      toolChunk([
        { index: 0, id: 'call_a', name: 'tool_a', arguments: '{"a":1}' },
        { index: 1, id: 'call_b', name: 'tool_b', arguments: '{"b":2}' },
      ]),
      toolChunk([], 'tool_calls'),
    ])
    const events = await collect(adapter)
    expect(events).toEqual([
      { type: 'tool_call_start', data: { toolCallId: 'call_a', name: 'tool_a' } },
      { type: 'tool_call_delta', data: { toolCallId: 'call_a', delta: '{"a":1}' } },
      { type: 'tool_call_start', data: { toolCallId: 'call_b', name: 'tool_b' } },
      { type: 'tool_call_delta', data: { toolCallId: 'call_b', delta: '{"b":2}' } },
      { type: 'tool_call_done', data: { toolCallId: 'call_a', input: { a: 1 } } },
      { type: 'tool_call_done', data: { toolCallId: 'call_b', input: { b: 2 } } },
    ])
  })

  test('5a. usage chunk (empty choices) → usage event', async () => {
    const { adapter } = adapterWithChunks([
      usageChunk({ prompt_tokens: 100, completion_tokens: 50 }),
    ])
    const events = await collect(adapter)
    expect(events).toEqual([
      { type: 'usage', data: { inputTokens: 100, outputTokens: 50 } },
    ])
  })

  test('5b. usage chunk with cached_tokens → cacheReadTokens populated', async () => {
    const { adapter } = adapterWithChunks([
      usageChunk({
        prompt_tokens:         100,
        completion_tokens:     50,
        prompt_tokens_details: { cached_tokens: 80 },
      }),
    ])
    const events = await collect(adapter)
    expect(events).toEqual([
      { type: 'usage', data: { inputTokens: 100, outputTokens: 50, cacheReadTokens: 80 } },
    ])
  })

  test('5c. cached_tokens === 0 → still populated (legitimate zero)', async () => {
    const { adapter } = adapterWithChunks([
      usageChunk({
        prompt_tokens:         100,
        completion_tokens:     50,
        prompt_tokens_details: { cached_tokens: 0 },
      }),
    ])
    const events = await collect(adapter)
    expect(events).toEqual([
      { type: 'usage', data: { inputTokens: 100, outputTokens: 50, cacheReadTokens: 0 } },
    ])
  })

  test('6. tool_call without id → falls back to idx-<index>, still start/done', async () => {
    const { adapter } = adapterWithChunks([
      toolChunk([{ index: 0, name: 'no_id_tool', arguments: '{"k":1}' }]),
      toolChunk([], 'tool_calls'),
    ])
    const events = await collect(adapter)
    expect(events).toEqual([
      { type: 'tool_call_start', data: { toolCallId: 'idx-0', name: 'no_id_tool' } },
      { type: 'tool_call_delta', data: { toolCallId: 'idx-0', delta: '{"k":1}' } },
      { type: 'tool_call_done', data: { toolCallId: 'idx-0', input: { k: 1 } } },
    ])
  })

  test('7. invalid JSON arguments → tool_call_done input = {} (no throw)', async () => {
    const { adapter } = adapterWithChunks([
      toolChunk([{ index: 0, id: 'call_bad', name: 'broken', arguments: 'not json{' }]),
      toolChunk([], 'tool_calls'),
    ])
    const events = await collect(adapter)
    expect(events).toContainEqual({
      type: 'tool_call_start',
      data: { toolCallId: 'call_bad', name: 'broken' },
    })
    expect(events).toContainEqual({
      type: 'tool_call_done',
      data: { toolCallId: 'call_bad', input: {} },
    })
  })

  test('7b. empty arguments buffer → tool_call_done input = {}', async () => {
    const { adapter } = adapterWithChunks([
      toolChunk([{ index: 0, id: 'call_empty', name: 'noargs' }]),
      toolChunk([], 'tool_calls'),
    ])
    const events = await collect(adapter)
    expect(events).toEqual([
      { type: 'tool_call_start', data: { toolCallId: 'call_empty', name: 'noargs' } },
      { type: 'tool_call_done', data: { toolCallId: 'call_empty', input: {} } },
    ])
  })

  test('8. empty stream (no chunks) → no events, no throw', async () => {
    const { adapter } = adapterWithChunks([])
    const events = await collect(adapter)
    expect(events).toEqual([])
  })

  test('finish_reason with empty Map → no spurious done events', async () => {
    const { adapter } = adapterWithChunks([
      contentChunk('done'),
      contentChunk(''),
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ])
    const events = await collect(adapter)
    // empty-string content yields nothing; finish with empty Map yields nothing
    expect(events).toEqual([{ type: 'message_delta', data: { text: 'done' } }])
  })

  test('protocol assumption: id on first chunk only — subsequent chunks without id do not overwrite toolCallId', async () => {
    // OpenAI protocol: id appears only in the first delta for a given index.
    // Subsequent deltas carry only arguments. This test locks that assumption:
    // the done event must use the real id from the first chunk, not a fallback.
    const { adapter } = adapterWithChunks([
      toolChunk([{ index: 0, id: 'real_id_abc', name: 'search', arguments: '{"q":' }]),
      // second chunk: same index, no id, no name — only arguments fragment
      toolChunk([{ index: 0, arguments: '"hello"}' }]),
      toolChunk([], 'tool_calls'),
    ])
    const events = await collect(adapter)
    expect(events).toEqual([
      { type: 'tool_call_start', data: { toolCallId: 'real_id_abc', name: 'search' } },
      { type: 'tool_call_delta', data: { toolCallId: 'real_id_abc', delta: '{"q":' } },
      { type: 'tool_call_delta', data: { toolCallId: 'real_id_abc', delta: '"hello"}' } },
      { type: 'tool_call_done', data: { toolCallId: 'real_id_abc', input: { q: 'hello' } } },
    ])
  })

  test('full sequence: text + tool_call + usage in one stream', async () => {
    const { adapter } = adapterWithChunks([
      contentChunk('Sure. '),
      toolChunk([{ index: 0, id: 'c1', name: 'f', arguments: '{"x":1}' }]),
      toolChunk([], 'tool_calls'),
      usageChunk({
        prompt_tokens:         10,
        completion_tokens:     5,
        prompt_tokens_details: { cached_tokens: 3 },
      }),
    ])
    const events = await collect(adapter)
    expect(events).toEqual([
      { type: 'message_delta', data: { text: 'Sure. ' } },
      { type: 'tool_call_start', data: { toolCallId: 'c1', name: 'f' } },
      { type: 'tool_call_delta', data: { toolCallId: 'c1', delta: '{"x":1}' } },
      { type: 'tool_call_done', data: { toolCallId: 'c1', input: { x: 1 } } },
      { type: 'usage', data: { inputTokens: 10, outputTokens: 5, cacheReadTokens: 3 } },
    ])
  })
})
