import { AnthropicAdapter } from '../gateway/AnthropicAdapter'
import type { ModelRequest, ModelResponse } from '../types/model'

// buildParams is private. We exercise it via cast — no network call.
function buildParamsOf(adapter: AnthropicAdapter, request: ModelRequest): Record<string, unknown> {
  return (adapter as unknown as { buildParams(r: ModelRequest): Record<string, unknown> }).buildParams(request)
}

// parseResponse is private. Same trick — feed it the raw shape Anthropic returns.
function parseResponseOf(adapter: AnthropicAdapter, raw: unknown): ModelResponse {
  return (adapter as unknown as { parseResponse(r: unknown): ModelResponse }).parseResponse(raw)
}

describe('AnthropicAdapter — cacheBreakpoint translation', () => {
  const adapter = new AnthropicAdapter({ apiKey: 'sk-test' })

  test('no cacheBreakpoint → system stays as string', () => {
    const params = buildParamsOf(adapter, {
      model:    'claude-sonnet-4-6',
      system:   'You are an agent.',
      messages: [],
    })
    expect(params['system']).toBe('You are an agent.')
  })

  test('cacheBreakpoint=system-end → system becomes block array with cache_control', () => {
    const params = buildParamsOf(adapter, {
      model:           'claude-sonnet-4-6',
      system:          'You are an agent. + persistent skills...',
      messages:        [],
      cacheBreakpoint: 'system-end',
    })
    expect(params['system']).toEqual([
      {
        type:          'text',
        text:          'You are an agent. + persistent skills...',
        cache_control: { type: 'ephemeral' },
      },
    ])
  })

  test('cacheBreakpoint=system-end with empty system → no system field emitted', () => {
    const params = buildParamsOf(adapter, {
      model:           'claude-sonnet-4-6',
      messages:        [],
      cacheBreakpoint: 'system-end',
    })
    expect(params['system']).toBeUndefined()
  })

  // #126: temperature passthrough — alfred's compressor calls with fast=True at a
  // fixed temperature; the adapter must forward it to Anthropic's messages.create.
  test('temperature is forwarded into params when set', () => {
    const params = buildParamsOf(adapter, {
      model:       'claude-sonnet-4-6',
      messages:    [],
      temperature: 0.2,
    })
    expect(params['temperature']).toBe(0.2)
  })

  test('temperature absent → no temperature field emitted (provider default)', () => {
    const params = buildParamsOf(adapter, {
      model:    'claude-sonnet-4-6',
      messages: [],
    })
    expect(params['temperature']).toBeUndefined()
  })

  test('cacheBreakpoint plays nicely with tools (both fields emitted)', () => {
    const params = buildParamsOf(adapter, {
      model:           'claude-sonnet-4-6',
      system:          'sys',
      messages:        [],
      tools:           [{ name: 'echo', description: 'e', inputSchema: {} }],
      cacheBreakpoint: 'system-end',
    })
    expect(params['system']).toEqual([
      { type: 'text', text: 'sys', cache_control: { type: 'ephemeral' } },
    ])
    expect(params['tools']).toHaveLength(1)
  })
})

describe('AnthropicAdapter — response usage extraction', () => {
  const adapter = new AnthropicAdapter({ apiKey: 'sk-test' })

  // Base shape modeled on a real anthropic Message response.
  const baseRaw = {
    content: [{ type: 'text' as const, text: 'hi' }],
    stop_reason: 'end_turn' as const,
  }

  test('extracts input + output tokens (no cache fields)', () => {
    const resp = parseResponseOf(adapter, {
      ...baseRaw,
      usage: { input_tokens: 1000, output_tokens: 50 },
    })
    expect(resp.usage).toEqual({ inputTokens: 1000, outputTokens: 50 })
  })

  // Regression: AnthropicAdapter previously read only input/output tokens and
  // dropped cache_read_input_tokens / cache_creation_input_tokens. As a result
  // even when the request used cacheBreakpoint:'system-end' and Anthropic
  // honored it, the trace's cacheStats showed up as undefined — the UI badge
  // had no way to surface "cache substrate engaged". These two tests assert
  // both numbers reach ModelUsage so the trace pipeline can build cacheStats.
  test('extracts cache_read_input_tokens into cacheReadTokens', () => {
    const resp = parseResponseOf(adapter, {
      ...baseRaw,
      usage: {
        input_tokens:              1200,
        output_tokens:             80,
        cache_read_input_tokens:   800,
      },
    })
    expect(resp.usage).toEqual({
      inputTokens:      1200,
      outputTokens:     80,
      cacheReadTokens:  800,
    })
  })

  test('extracts cache_creation_input_tokens into cacheCreationTokens', () => {
    const resp = parseResponseOf(adapter, {
      ...baseRaw,
      usage: {
        input_tokens:                  1500,
        output_tokens:                 60,
        cache_creation_input_tokens:   750,
      },
    })
    expect(resp.usage).toEqual({
      inputTokens:          1500,
      outputTokens:         60,
      cacheCreationTokens:  750,
    })
  })

  test('extracts both cache fields when present in same response', () => {
    const resp = parseResponseOf(adapter, {
      ...baseRaw,
      usage: {
        input_tokens:                 2000,
        output_tokens:                100,
        cache_read_input_tokens:      1200,
        cache_creation_input_tokens:  400,
      },
    })
    expect(resp.usage).toEqual({
      inputTokens:          2000,
      outputTokens:         100,
      cacheReadTokens:      1200,
      cacheCreationTokens:  400,
    })
  })

  test('treats null cache_*_input_tokens as absent', () => {
    // Anthropic returns null (not 0) for these fields when the request had
    // no cache_control marker. We should NOT report them as 0 — that would
    // bias hit rate calculations and signal "cache subsystem engaged" when
    // it wasn't. Absent === not engaged.
    const resp = parseResponseOf(adapter, {
      ...baseRaw,
      usage: {
        input_tokens:                  500,
        output_tokens:                 30,
        cache_read_input_tokens:       null,
        cache_creation_input_tokens:   null,
      },
    })
    expect(resp.usage).toEqual({ inputTokens: 500, outputTokens: 30 })
  })
})
