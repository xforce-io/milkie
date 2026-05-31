import { explainToolCall } from '../trace/diagnostics/explainToolCall'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, payload: unknown, causedBy?: string): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload, ...(causedBy ? { causedBy } : {}) })

function scenario(): Event[] {
  return [
    ev('start', 'agent.run.started', {}),
    ev('lr1', 'llm.responded', {}, 'start'),
    ev('treq', 'tool.requested', { toolName: 'search', input: { q: 'x' }, requestHash: 'h' }, 'lr1'),
    ev('tres', 'tool.responded', { toolName: 'search', output: { hits: 3 }, requestHash: 'h' }, 'treq'),
  ]
}

describe('explainToolCall', () => {
  it('projects toolName, input, paired output, trigger and causal chain', () => {
    const exp = explainToolCall(scenario(), 'treq')
    expect(exp.toolName).toBe('search')
    expect(exp.input).toEqual({ q: 'x' })
    expect(exp.output).toEqual({ hits: 3 })
    expect(exp.trigger.causedByEventId).toBe('lr1')
    expect(exp.trigger.causedBySummary).toBe('llm.responded')
    expect(exp.causalChain.map(c => c.eventId)).toEqual(['treq', 'lr1', 'start'])
    expect(exp.summary).toContain('search')
  })

  it('omits output when there is no paired tool.responded', () => {
    const events: Event[] = [ev('treq', 'tool.requested', { toolName: 'search', input: {} })]
    expect(explainToolCall(events, 'treq').output).toBeUndefined()
  })

  it('throws on a non-tool.requested event id', () => {
    expect(() => explainToolCall(scenario(), 'tres')).toThrow(/tool\.requested/)
  })
})
