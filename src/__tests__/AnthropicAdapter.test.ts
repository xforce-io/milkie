import { AnthropicAdapter } from '../gateway/AnthropicAdapter'
import type { ModelRequest } from '../types/model'

// buildParams is private. We exercise it via cast — no network call.
function buildParamsOf(adapter: AnthropicAdapter, request: ModelRequest): Record<string, unknown> {
  return (adapter as unknown as { buildParams(r: ModelRequest): Record<string, unknown> }).buildParams(request)
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
