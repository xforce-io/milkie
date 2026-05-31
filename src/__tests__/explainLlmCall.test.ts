import { explainLlmCall } from '../trace/diagnostics/explainLlmCall'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, payload: unknown, causedBy?: string): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload, ...(causedBy ? { causedBy } : {}) })
const region = (id: string, contentHash?: string) =>
  ev(`add-${id}`, 'region.added', { id, target: 'message', section: 's', stability: 'volatile', reason: 'r', ...(contentHash ? { contentHash } : {}) })

function scenario(): Event[] {
  return [
    ev('start', 'agent.run.started', { agentId: 'a', goal: 'g', input: 'i', contextId: 'c' }),
    region('header', 'H1'),
    region('hist', 'H2'),
    ev('treq', 'tool.requested', { toolName: 'search', input: {} }, 'start'),
    ev('tres', 'tool.responded', { toolName: 'search', output: {} }, 'treq'),
    ev('fsm', 'fsm.transition', { from: 'plan', to: 'reflect', trigger: { domain: 'business', name: 'NEXT' } }, 'tres'),
    ev('llm', 'llm.requested', { model: 'm' }, 'tres'),
  ]
}

describe('explainLlmCall', () => {
  it('projects trigger, fsm state, region count, causal chain and summary (no LLM)', () => {
    const exp = explainLlmCall(scenario(), 'llm')
    expect(exp.trigger.causedByEventId).toBe('tres')
    expect(exp.trigger.causedBySummary).toBe('tool.responded(search)')
    expect(exp.fsmState).toBe('reflect')
    expect(exp.regionCount).toBe(2)
    expect(exp.causalChain.map(c => c.eventId)).toEqual(['llm', 'tres', 'treq', 'start'])
    expect(exp.summary).toContain('reflect')
    expect(exp.summary).toContain('tool.responded(search)')
    expect(exp.summary).toContain('2')
  })

  it('falls back when there is no upstream trigger and no transitions', () => {
    const events: Event[] = [ev('llm', 'llm.requested', { model: 'm' })]
    const exp = explainLlmCall(events, 'llm')
    expect(exp.trigger.causedByEventId).toBeUndefined()
    expect(exp.fsmState).toBeNull()
    expect(exp.regionCount).toBe(0)
    expect(exp.summary).toContain('(无上游记录)')
  })

  it('handles a dangling causedBy: keeps causedByEventId, omits causedBySummary, falls back in summary', () => {
    const events: Event[] = [ev('llm', 'llm.requested', { model: 'm' }, 'missing')]
    const exp = explainLlmCall(events, 'llm')
    expect(exp.trigger.causedByEventId).toBe('missing')
    expect(exp.trigger.causedBySummary).toBeUndefined()
    expect(exp.summary).toContain('(无上游记录)')
    expect(exp.causalChain.map(c => c.eventId)).toEqual(['llm'])
  })

  it('throws on a non-llm.requested event id', () => {
    expect(() => explainLlmCall(scenario(), 'fsm')).toThrow(/llm\.requested/)
  })

  it('throws on an unknown event id', () => {
    expect(() => explainLlmCall(scenario(), 'nope')).toThrow(/nope/)
  })
})
