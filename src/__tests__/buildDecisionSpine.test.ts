import { buildDecisionSpine } from '../trace/diagnostics/buildDecisionSpine'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, payload: unknown, causedBy?: string, ts = 0): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: ts, payload, ...(causedBy ? { causedBy } : {}) })

// #175 de-core: a decision node is an I/O EFFECT (llm.responded / tool.responded)
// plus the final output — never an `fsm.transition` business-topology node.
// llm.requested / tool.requested are the *triggers*; the spine anchors on the
// responded effects, and causeDecisionId walks causedBy to the nearest effect.
function scenario(): Event[] {
  return [
    ev('start', 'agent.run.started', { agentId: 'a', goal: 'g', input: 'i', contextId: 'c' }, undefined, 1),
    ev('llm1', 'llm.requested', { model: 'm' }, 'start', 2),
    ev('lr1', 'llm.responded', { response: { content: [], toolCalls: [{ name: 'classify_intent', input: {} }] }, requestHash: 'h1' }, 'llm1', 3),
    ev('treq', 'tool.requested', { toolName: 'classify_intent', input: {} }, 'lr1', 4),
    ev('tres', 'tool.responded', { toolName: 'classify_intent', output: {} }, 'treq', 5),
    ev('done', 'agent.run.completed', { status: 'completed', lastTextOutput: 'ok' }, 'tres', 7),
  ]
}

describe('buildDecisionSpine', () => {
  it('keeps only decision-effect nodes in timestamp order with labels', () => {
    const spine = buildDecisionSpine(scenario())
    expect(spine.nodes.map(n => [n.kind, n.eventId])).toEqual([
      ['llm', 'lr1'], ['tool', 'tres'], ['output', 'done'],
    ])
    expect(spine.nodes.find(n => n.eventId === 'tres')!.label).toBe('tool: classify_intent')
    expect(spine.nodes.find(n => n.eventId === 'lr1')!.label).toBe('LLM → classify_intent')
  })

  it('resolves causeDecisionId to the nearest decision ancestor (skipping non-decision causes)', () => {
    const spine = buildDecisionSpine(scenario())
    // tool.responded -> tool.requested -> llm.responded(lr1): nearest effect ancestor is lr1
    expect(spine.nodes.find(n => n.eventId === 'tres')!.causeDecisionId).toBe('lr1')
    // output -> tool.responded(tres) directly
    expect(spine.nodes.find(n => n.eventId === 'done')!.causeDecisionId).toBe('tres')
    // first llm.responded only chains up through llm.requested -> agent.run.started (no effect ancestor)
    expect(spine.nodes.find(n => n.eventId === 'lr1')!.causeDecisionId).toBeUndefined()
  })

  it('returns empty spine for a run with no decisions', () => {
    expect(buildDecisionSpine([ev('s', 'agent.run.started', {})]).nodes).toEqual([])
  })

  it('labels v2 failure terminals as LLM failure · code, not text success', () => {
    const events: Event[] = [
      ev('llm1', 'llm.requested', { requestHash: 'h', outcomeSchemaVersion: 2 }, undefined, 1),
      ev('lr1', 'llm.responded', {
        status: 'error',
        requestHash: 'h',
        error: {
          code: 'MODEL_RATE_LIMITED',
          message: 'Model provider rate limit exceeded.',
          phase: 'request',
          provider: 'anthropic',
          model: 'm',
          retryable: true,
        },
      }, 'llm1', 2),
    ]
    const node = buildDecisionSpine(events).nodes.find(n => n.eventId === 'lr1')!
    expect(node.label).toBe('LLM failure · MODEL_RATE_LIMITED')
    expect(node.status).toBe('error')
    expect(node.label).not.toBe('LLM → 文本')
  })
})
