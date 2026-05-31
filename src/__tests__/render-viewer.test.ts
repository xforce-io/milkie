import { renderViewer } from '../trace/render/viewer'
import type { Event } from '../trace/types'

const e = (over: Partial<Event> & { id: string; runId: string; type: Event['type'] }): Event =>
  ({ actor: 'a', timestamp: 0, payload: {}, ...over })

function scenario(): Event[] {
  return [
    e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1, payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
    e({ id: 'llm1', runId: 'r1', type: 'llm.requested', timestamp: 2, causedBy: 'start', payload: { model: 'm' } }),
    e({ id: 'lr1', runId: 'r1', type: 'llm.responded', timestamp: 3, causedBy: 'llm1', payload: {} }),
    e({ id: 'treq', runId: 'r1', type: 'tool.requested', timestamp: 4, causedBy: 'lr1', payload: { toolName: 'classify_intent', input: {}, requestHash: 'h' } }),
    e({ id: 'tres', runId: 'r1', type: 'tool.responded', timestamp: 5, causedBy: 'treq', payload: { toolName: 'classify_intent', output: {}, requestHash: 'h' } }),
    e({ id: 'fsm', runId: 'r1', type: 'fsm.transition', timestamp: 6, causedBy: 'tres', payload: { from: 'classify', to: 'handle', trigger: { domain: 'business', name: 'GO' }, guardEvaluations: [{ guardId: 'g1', result: 'GO', contextSlice: {} }] } }),
    e({ id: 'done', runId: 'r1', type: 'agent.run.completed', timestamp: 7, causedBy: 'fsm', payload: { status: 'completed', lastTextOutput: 'ok' } }),
  ]
}

describe('renderViewer', () => {
  it('produces a self-contained document with a decision spine and embedded explanations', () => {
    const html = renderViewer(scenario())
    expect(html.startsWith('<!doctype html>')).toBe(true)
    // spine has nodes with data-id for each decision
    expect(html).toContain('data-id="llm1"')
    expect(html).toContain('data-id="treq"')
    expect(html).toContain('data-id="fsm"')
    expect(html).toContain('data-id="done"')
    // output node carries the Why entry
    expect(html).toContain('class="spine-output"')
    // embedded data the JS reads
    expect(html).toContain('id="spine-data"')
    expect(html).toContain('id="explanations-data"')
    // why panel container + the decision/raw tabs
    expect(html).toContain('id="why-panel"')
    expect(html).toContain('data-tab="decision"')
    expect(html).toContain('data-tab="raw"')
    // raw tab reuses the timeline (filters present)
    expect(html).toContain('class="filters"')
  })

  it('renders without crashing for a run with no decisions', () => {
    const html = renderViewer([{ id: 's', runId: 'r1', actor: 'a', type: 'agent.run.started', timestamp: 1, payload: {} } as Event])
    expect(html.startsWith('<!doctype html>')).toBe(true)
    expect(html).toContain('id="why-panel"')
  })
})
