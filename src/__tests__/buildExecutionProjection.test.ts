import { buildExecutionProjection } from '../trace/diagnostics/buildExecutionProjection'
import type { Event } from '../trace/types'

// Minimal event builders — these are the recorded-log shapes the projection reads.
function llmRequested(id: string, hash: string, messages: unknown[] = [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }]): Event {
  return { id, runId: 'r', type: 'llm.requested', actor: 'a', timestamp: 1,
    payload: { request: { model: 'm', messages }, requestHash: hash } }
}
function llmResponded(id: string, hash: string, cacheStats?: unknown): Event {
  return { id, runId: 'r', type: 'llm.responded', actor: 'a', timestamp: 2, causedBy: hash,
    payload: { response: { content: [], toolCalls: [], finishReason: 'end_turn' }, requestHash: hash,
      ...(cacheStats ? { cacheStats } : {}) } }
}
function regionAdded(id: string, stability: string, contentHash?: string): Event {
  return { id: `ra-${id}`, runId: 'r', type: 'region.added', actor: 'a', timestamp: 0,
    payload: { id, target: 'message', section: 'sec', stability, reason: 'agent-set',
      ...(contentHash ? { contentHash } : {}) } }
}
function toolRequested(id: string, hash: string, name: string, input: unknown = {}): Event {
  return { id, runId: 'r', type: 'tool.requested', actor: 'a', timestamp: 3,
    payload: { toolName: name, input, requestHash: hash } }
}
function toolResponded(id: string, hash: string, name: string, body: { output?: unknown; error?: unknown }): Event {
  return { id, runId: 'r', type: 'tool.responded', actor: 'a', timestamp: 4, causedBy: hash,
    payload: { toolName: name, requestHash: hash, ...body } }
}

describe('buildExecutionProjection', () => {
  it('classifies a high-hit-rate llm call as a hot cache tier', () => {
    const events: Event[] = [
      llmRequested('llm-req', 'h1'),
      llmResponded('llm-resp', 'h1', { readTokens: 90, creationTokens: 0, totalInputTokens: 100, hitRate: 0.9 }),
    ]
    const proj = buildExecutionProjection(events)
    expect(proj.steps).toHaveLength(1)
    expect(proj.steps[0]!.kind).toBe('llm')
    expect(proj.steps[0]!.cacheHealth?.tier).toBe('hot')
  })

  it('classifies partial reuse or a fresh cache write as warm', () => {
    const partialRead = buildExecutionProjection([
      llmRequested('q1', 'h1'),
      llmResponded('r1', 'h1', { readTokens: 40, creationTokens: 0, totalInputTokens: 100, hitRate: 0.4 }),
    ])
    expect(partialRead.steps[0]!.cacheHealth?.tier).toBe('warm')

    const freshWrite = buildExecutionProjection([
      llmRequested('q2', 'h2'),
      llmResponded('r2', 'h2', { readTokens: 0, creationTokens: 80, totalInputTokens: 100, hitRate: 0 }),
    ])
    expect(freshWrite.steps[0]!.cacheHealth?.tier).toBe('warm')
  })

  it('classifies no-read no-write as cold', () => {
    const proj = buildExecutionProjection([
      llmRequested('q', 'h'),
      llmResponded('r', 'h', { readTokens: 0, creationTokens: 0, totalInputTokens: 100, hitRate: 0 }),
    ])
    expect(proj.steps[0]!.cacheHealth?.tier).toBe('cold')
  })

  it('reports null cacheHealth when the response carries no cacheStats', () => {
    const proj = buildExecutionProjection([llmRequested('q', 'h'), llmResponded('r', 'h')])
    expect(proj.steps[0]!.cacheHealth).toBeNull()
  })

  it('groups regions in context by stability in canonical order, omitting empty groups', () => {
    const events: Event[] = [
      regionAdded('header', 'immutable'),
      regionAdded('scratch', 'volatile'),
      regionAdded('skill', 'session-stable'),
      llmRequested('llm-req', 'h1'),
      llmResponded('llm-resp', 'h1'),
    ]
    const groups = buildExecutionProjection(events).steps[0]!.regionGroups!
    // immutable → session-stable → turn-stable → volatile; turn-stable absent → omitted
    expect(groups.map(g => g.stability)).toEqual(['immutable', 'session-stable', 'volatile'])
    expect(groups[0]!.regions.map(r => r.id)).toEqual(['header'])
  })

  it('emits tool steps with ok / error / pending status, in event order', () => {
    const events: Event[] = [
      toolRequested('t1', 'ha', 'search', { q: 'x' }),
      toolResponded('t1r', 'ha', 'search', { output: { hits: 3 } }),
      toolRequested('t2', 'hb', 'fetch'),
      toolResponded('t2r', 'hb', 'fetch', { error: { message: 'boom' } }),
      toolRequested('t3', 'hc', 'slow'),  // no response → pending
    ]
    const steps = buildExecutionProjection(events).steps
    expect(steps.map(s => s.kind)).toEqual(['tool', 'tool', 'tool'])
    expect(steps.map(s => s.tool?.status)).toEqual(['ok', 'error', 'pending'])
    expect(steps[0]!.tool).toMatchObject({ name: 'search', input: { q: 'x' }, output: { hits: 3 } })
    expect(steps[0]!.label).toBe('Tool · search')
    expect(steps[1]!.tool?.error).toMatchObject({ message: 'boom' })
  })

  it('preserves event order across interleaved llm and tool steps', () => {
    const events: Event[] = [
      llmRequested('q1', 'h1'),
      llmResponded('r1', 'h1'),
      toolRequested('t1', 'ha', 'search'),
      toolResponded('t1r', 'ha', 'search', { output: {} }),
      llmRequested('q2', 'h2'),
      llmResponded('r2', 'h2'),
    ]
    expect(buildExecutionProjection(events).steps.map(s => s.kind)).toEqual(['llm', 'tool', 'llm'])
  })

  it('fills region content from the regionContent map by contentHash, degrading when absent', () => {
    const events: Event[] = [
      regionAdded('header', 'immutable', 'ch1'),
      regionAdded('scratch', 'volatile'),  // no contentHash → no content
      llmRequested('q', 'h1'),
      llmResponded('r', 'h1'),
    ]
    const proj = buildExecutionProjection(events, { regionContent: new Map([['ch1', 'SYSTEM PROMPT TEXT']]) })
    const regions = proj.steps[0]!.regionGroups!.flatMap(g => g.regions)
    expect(regions.find(r => r.id === 'header')!.content).toBe('SYSTEM PROMPT TEXT')
    expect(regions.find(r => r.id === 'scratch')!.content).toBeUndefined()
  })

  it('carries message count, prompt, and response on the llm step', () => {
    const messages = [
      { role: 'user', content: [{ type: 'text', text: 'hi' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'yo' }] },
    ]
    const events: Event[] = [
      { id: 'q', runId: 'r', type: 'llm.requested', actor: 'a', timestamp: 1,
        payload: { request: { model: 'm', system: 'SYS', messages, tools: [{ name: 'search' }] }, requestHash: 'h1' } },
      { id: 'resp', runId: 'r', type: 'llm.responded', actor: 'a', timestamp: 2, causedBy: 'h1',
        payload: { response: { content: [{ type: 'text', text: 'hello' }], toolCalls: [], finishReason: 'end_turn' }, requestHash: 'h1' } },
    ]
    const step = buildExecutionProjection(events).steps[0]!
    expect(step.messageCount).toBe(2)
    expect(step.prompt).toMatchObject({ system: 'SYS', messages, tools: [{ name: 'search' }] })
    expect(step.response).toMatchObject({ content: [{ type: 'text', text: 'hello' }] })
  })
})
