import { OpenAICompatibleAdapter } from '../gateway/OpenAICompatibleAdapter'
import { DefaultIOPort } from '../runtime/IOPort'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { RecordingIOPort } from '../trace/RecordingIOPort'
import type { ModelRequest, ModelResponse } from '../types/model'

// parseResponse is private; we exercise it via cast — no network call.
function parseResponseOf(adapter: OpenAICompatibleAdapter, raw: unknown): ModelResponse {
  return (adapter as unknown as { parseResponse(r: unknown): ModelResponse }).parseResponse(raw)
}

// Stub the underlying OpenAI client's chat.completions.create to capture the
// params the adapter builds — no network. Returns a minimal completion shape.
function stubCreate(adapter: OpenAICompatibleAdapter): { calls: unknown[] } {
  const calls: unknown[] = []
  const create = async (params: unknown): Promise<unknown> => {
    calls.push(params)
    return { choices: [{ message: { role: 'assistant', content: 'hi' }, finish_reason: 'stop' }] }
  }
  ;(adapter as unknown as { client: { chat: { completions: { create: unknown } } } })
    .client.chat.completions.create = create
  return { calls }
}

describe('OpenAICompatibleAdapter — cache stats extraction', () => {
  const adapter = new OpenAICompatibleAdapter({ apiKey: 'sk-test' })

  test('no usage → no usage on ModelResponse', () => {
    const raw = {
      choices: [{ message: { role: 'assistant', content: 'hi' }, finish_reason: 'stop' }],
    }
    const out = parseResponseOf(adapter, raw)
    expect(out.usage).toBeUndefined()
  })

  test('usage without prompt_tokens_details → cacheReadTokens absent', () => {
    const raw = {
      choices: [{ message: { role: 'assistant', content: 'hi' }, finish_reason: 'stop' }],
      usage:   { prompt_tokens: 100, completion_tokens: 50 },
    }
    const out = parseResponseOf(adapter, raw)
    expect(out.usage).toEqual({ inputTokens: 100, outputTokens: 50 })
    expect(out.usage?.cacheReadTokens).toBeUndefined()
  })

  test('usage with prompt_tokens_details.cached_tokens → cacheReadTokens populated', () => {
    const raw = {
      choices: [{ message: { role: 'assistant', content: 'hi' }, finish_reason: 'stop' }],
      usage:   {
        prompt_tokens:         100,
        completion_tokens:     50,
        prompt_tokens_details: { cached_tokens: 80 },
      },
    }
    const out = parseResponseOf(adapter, raw)
    expect(out.usage).toEqual({
      inputTokens:     100,
      outputTokens:    50,
      cacheReadTokens: 80,
    })
  })

  test('prompt_tokens_details.cached_tokens === 0 → still populated (legitimate zero, not absence)', () => {
    const raw = {
      choices: [{ message: { role: 'assistant', content: 'hi' }, finish_reason: 'stop' }],
      usage:   {
        prompt_tokens:         100,
        completion_tokens:     50,
        prompt_tokens_details: { cached_tokens: 0 },
      },
    }
    const out = parseResponseOf(adapter, raw)
    expect(out.usage?.cacheReadTokens).toBe(0)
  })
})

describe('OpenAICompatibleAdapter — invalid tool arguments', () => {
  test('preserves malformed arguments metadata while valid empty object remains valid', () => {
    const adapter = new OpenAICompatibleAdapter({ apiKey: 'sk-test' })
    const response = parseResponseOf(adapter, {
      choices: [{
        message: {
          role: 'assistant',
          tool_calls: [{
            id:       'call-invalid',
            type:     'function',
            function: { name: 'search', arguments: 'secret-malformed-token' },
          }],
        },
        finish_reason: 'tool_calls',
      }],
    })
    const validEmpty = parseResponseOf(adapter, {
      choices: [{
        message: {
          role: 'assistant',
          tool_calls: [{
            id:       'call-empty',
            type:     'function',
            function: { name: 'search', arguments: '{}' },
          }],
        },
        finish_reason: 'tool_calls',
      }],
    })

    expect(response.toolCalls[0]).toMatchObject({
      input: {},
      invalidArguments: {
        code:      'TOOL_ARGUMENTS_INVALID_JSON',
        message:   'Tool arguments are not valid JSON',
        rawLength: 'secret-malformed-token'.length,
      },
    })
    expect(response.content[0]).toMatchObject({
      type: 'tool_use',
      input: {},
      invalidArguments: { code: 'TOOL_ARGUMENTS_INVALID_JSON' },
    })
    expect(validEmpty.toolCalls[0]).not.toHaveProperty('invalidArguments')
    expect(response.raw).toBeUndefined()
  })
  test('marks missing function arguments invalid with a safe zero raw length', () => {
    const adapter = new OpenAICompatibleAdapter({ apiKey: 'sk-test' })
    const response = parseResponseOf(adapter, {
      choices: [{
        message: {
          role: 'assistant',
          tool_calls: [{
            id:       'call-missing',
            type:     'function',
            function: { name: 'search' },
          }],
        },
        finish_reason: 'tool_calls',
      }],
    })

    expect(response.toolCalls[0]).toMatchObject({
      input: {},
      invalidArguments: {
        code:      'TOOL_ARGUMENTS_INVALID_JSON',
        message:   'Tool arguments are not valid JSON',
        rawLength: 0,
      },
    })
  })
  test('does not record malformed provider arguments in llm.responded', async () => {
    const adapter = new OpenAICompatibleAdapter({ apiKey: 'sk-test' })
    const response = parseResponseOf(adapter, {
      choices: [{
        message: {
          role: 'assistant',
          tool_calls: [{
            id:       'call-trace',
            type:     'function',
            function: { name: 'search', arguments: 'secret-trace-token' },
          }],
        },
        finish_reason: 'tool_calls',
      }],
    })
    const store = new MemoryEventStore()
    const port = new RecordingIOPort(new DefaultIOPort({
      async complete(): Promise<ModelResponse> { return response },
      async *stream(): AsyncIterable<never> { yield* [] },
    }), store, 'raw-trace')

    await port.invokeLLM({
      model:    'test',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
    })

    const responded = (await store.readByRunId('raw-trace'))
      .find(event => event.type === 'llm.responded')
    expect(JSON.stringify(responded?.payload)).not.toContain('secret-trace-token')
  })
})


describe('OpenAICompatibleAdapter — temperature passthrough (#126)', () => {
  const baseReq: ModelRequest = {
    model:    'qwen-turbo',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
  }

  test('complete forwards temperature to chat.completions.create when set', async () => {
    const adapter = new OpenAICompatibleAdapter({ apiKey: 'sk-test' })
    const { calls } = stubCreate(adapter)
    await adapter.complete({ ...baseReq, temperature: 0.2 })
    expect((calls[0] as { temperature?: number }).temperature).toBe(0.2)
  })

  test('complete omits temperature when not set (provider default)', async () => {
    const adapter = new OpenAICompatibleAdapter({ apiKey: 'sk-test' })
    const { calls } = stubCreate(adapter)
    await adapter.complete(baseReq)
    expect((calls[0] as { temperature?: number }).temperature).toBeUndefined()
  })

  test('stream forwards temperature to chat.completions.create when set', async () => {
    const adapter = new OpenAICompatibleAdapter({ apiKey: 'sk-test' })
    const calls: unknown[] = []
    const create = async (params: unknown): Promise<AsyncIterable<never>> => {
      calls.push(params)
      return (async function* () { /* no chunks */ })()
    }
    ;(adapter as unknown as { client: { chat: { completions: { create: unknown } } } })
      .client.chat.completions.create = create
    for await (const _e of adapter.stream({ ...baseReq, temperature: 0.7 })) { void _e }
    expect((calls[0] as { temperature?: number }).temperature).toBe(0.7)
  })
})

describe('OpenAICompatibleAdapter — structured failures (#202)', () => {
  const req: ModelRequest = {
    model: 'glm-5.2', messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
  }

  test('normalizes request failures with provider/model metadata', async () => {
    const adapter = new OpenAICompatibleAdapter({ apiKey: 'sk-test', provider: 'volcengine' })
    const cause = Object.assign(new Error('secret endpoint'), { name: 'APIConnectionError' })
    ;(adapter as unknown as { client: { chat: { completions: { create: unknown } } } })
      .client.chat.completions.create = async () => { throw cause }

    await expect(adapter.complete(req)).rejects.toMatchObject({
      envelope: {
        code: 'MODEL_CONNECTION_ERROR', message: 'Model provider connection failed.',
        phase: 'request', provider: 'volcengine', model: 'glm-5.2', retryable: true,
      },
      cause,
    })
  })

  test('classifies response parsing failures as bad responses', async () => {
    const adapter = new OpenAICompatibleAdapter({ apiKey: 'sk-test', provider: 'volcengine' })
    ;(adapter as unknown as { client: { chat: { completions: { create: unknown } } } })
      .client.chat.completions.create = async () => ({ choices: [] })

    await expect(adapter.complete(req)).rejects.toMatchObject({
      envelope: { code: 'MODEL_BAD_RESPONSE', phase: 'response_parse', retryable: false },
    })
  })
})
