import { buildDecisionSpine } from '../trace/diagnostics/buildDecisionSpine'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, payload: unknown, causedBy?: string, ts = 0): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: ts, payload, ...(causedBy ? { causedBy } : {}) })

function scenario(): Event[] {
  return [
    ev('start', 'agent.run.started', { agentId: 'a', goal: 'g', input: 'i', contextId: 'c' }, undefined, 1),
    ev('llm1', 'llm.requested', { model: 'm' }, 'start', 2),
    ev('lr1', 'llm.responded', {}, 'llm1', 3),
    ev('treq', 'tool.requested', { toolName: 'classify_intent', input: {} }, 'lr1', 4),
    ev('tres', 'tool.responded', { toolName: 'classify_intent', output: {} }, 'treq', 5),
    ev('fsm', 'fsm.transition', { from: 'classify', to: 'handle', trigger: { domain: 'business', name: 'GO' } }, 'tres', 6),
    ev('done', 'agent.run.completed', { status: 'completed', lastTextOutput: 'ok' }, 'fsm', 7),
  ]
}

describe('buildDecisionSpine', () => {
  it('keeps only decision nodes in timestamp order with labels', () => {
    const spine = buildDecisionSpine(scenario())
    expect(spine.nodes.map(n => [n.kind, n.eventId])).toEqual([
      ['llm', 'llm1'], ['tool', 'treq'], ['transition', 'fsm'], ['output', 'done'],
    ])
    expect(spine.nodes.find(n => n.eventId === 'treq')!.label).toBe('tool: classify_intent')
    expect(spine.nodes.find(n => n.eventId === 'fsm')!.label).toBe('classify → handle')
  })

  it('resolves causeDecisionId to the nearest decision ancestor (skipping non-decision causes)', () => {
    const spine = buildDecisionSpine(scenario())
    expect(spine.nodes.find(n => n.eventId === 'fsm')!.causeDecisionId).toBe('treq')
    expect(spine.nodes.find(n => n.eventId === 'done')!.causeDecisionId).toBe('fsm')
    expect(spine.nodes.find(n => n.eventId === 'llm1')!.causeDecisionId).toBeUndefined()
  })

  it('returns empty spine for a run with no decisions', () => {
    expect(buildDecisionSpine([ev('s', 'agent.run.started', {})]).nodes).toEqual([])
  })
})
