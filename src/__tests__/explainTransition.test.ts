import { explainTransition } from '../trace/diagnostics/explainTransition'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, payload: unknown, causedBy?: string): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload, ...(causedBy ? { causedBy } : {}) })

function routingEvents(): Event[] {
  return [
    ev('start', 'agent.run.started', { agentId: 'a', goal: 'g', input: 'i', contextId: 'c' }),
    ev('llm1', 'llm.responded', {}, 'start'),
    ev('treq', 'tool.requested', { toolName: 'classify_intent', input: {} }, 'llm1'),
    ev('tres', 'tool.responded', { toolName: 'classify_intent', output: {} }, 'treq'),
    ev('fsm', 'fsm.transition', {
      from: 'classify', to: 'handle_b',
      trigger: { domain: 'business', name: 'INTENT_B' },
      guardEvaluations: [{ guardId: 'intent-threshold', result: 'INTENT_B', contextSlice: { confidence: 0.9, threshold: 0.75 } }],
    }, 'tres'),
  ]
}

describe('explainTransition', () => {
  it('projects a readable explanation from the event log (no LLM)', () => {
    const exp = explainTransition(routingEvents(), 'fsm')
    expect(exp.from).toBe('classify')
    expect(exp.to).toBe('handle_b')
    expect(exp.trigger.name).toBe('INTENT_B')
    expect(exp.trigger.causedByEventId).toBe('tres')
    expect(exp.trigger.causedBySummary).toBe('tool.responded(classify_intent)')
    expect(exp.guards).toEqual([{ guardId: 'intent-threshold', result: 'INTENT_B', contextSlice: { confidence: 0.9, threshold: 0.75 } }])
    expect(exp.causalChain.map(c => c.eventId)).toEqual(['fsm', 'tres', 'treq', 'llm1', 'start'])
    expect(exp.summary).toContain('classify → handle_b')
    expect(exp.summary).toContain('INTENT_B')
    expect(exp.summary).toContain('intent-threshold')
  })

  it('every causalChain eventId resolves to a real event', () => {
    const events = routingEvents()
    const ids = new Set(events.map(e => e.id))
    const exp = explainTransition(events, 'fsm')
    for (const c of exp.causalChain) expect(ids.has(c.eventId)).toBe(true)
  })

  it('omits guards when the transition carries none', () => {
    const events: Event[] = [
      ev('start', 'agent.run.started', {}),
      ev('fsm', 'fsm.transition', { from: 's0', to: 'end', trigger: { domain: 'lifecycle', name: 'DONE' } }, 'start'),
    ]
    const exp = explainTransition(events, 'fsm')
    expect(exp.guards).toEqual([])
    expect(exp.summary).toContain('s0 → end')
  })

  it('throws on a non-fsm.transition event id', () => {
    expect(() => explainTransition(routingEvents(), 'llm1')).toThrow(/fsm\.transition/)
  })

  it('throws on an unknown event id', () => {
    expect(() => explainTransition(routingEvents(), 'nope')).toThrow(/nope/)
  })

  it('handles a dangling causedBy: keeps causedByEventId, omits causedBySummary, falls back in summary', () => {
    const events: Event[] = [
      ev('fsm', 'fsm.transition', { from: 'a', to: 'b', trigger: { domain: 'business', name: 'GO' } }, 'missing'),
    ]
    const exp = explainTransition(events, 'fsm')
    expect(exp.trigger.causedByEventId).toBe('missing')
    expect(exp.trigger.causedBySummary).toBeUndefined()
    expect(exp.summary).toContain('(无上游记录)')
    expect(exp.causalChain.map(c => c.eventId)).toEqual(['fsm'])  // walkCausedBy stops at the dangling ref
  })
})
