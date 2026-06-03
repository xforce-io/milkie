import { OpenAICompatibleAdapter } from '../gateway/OpenAICompatibleAdapter'
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
