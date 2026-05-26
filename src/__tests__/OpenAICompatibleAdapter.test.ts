import { OpenAICompatibleAdapter } from '../gateway/OpenAICompatibleAdapter'
import type { ModelResponse } from '../types/model'

// parseResponse is private; we exercise it via cast — no network call.
function parseResponseOf(adapter: OpenAICompatibleAdapter, raw: unknown): ModelResponse {
  return (adapter as unknown as { parseResponse(r: unknown): ModelResponse }).parseResponse(raw)
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
