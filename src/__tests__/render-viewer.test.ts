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

  it('embeds LLM region composition with content when regionContent is provided', () => {
    const events: Event[] = [
      e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1, payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 'h', runId: 'r1', type: 'region.added', timestamp: 2, payload: { id: 'header', target: 'system', section: 'header', stability: 'immutable', reason: 'agent-set', contentHash: 'H1' } }),
      e({ id: 'llm1', runId: 'r1', type: 'llm.requested', timestamp: 3, causedBy: 'start', payload: { model: 'm' } }),
    ]
    const html = renderViewer(events, { regionContent: new Map([['H1', 'SYSTEM PROMPT TEXT']]) })
    expect(html).toContain('Assembled by 1 regions')
    expect(html).toContain('header')
    expect(html).toContain('SYSTEM PROMPT TEXT')
  })

  it('keeps the inactive decision pane hidden so the decision/raw toggle works', () => {
    const html = renderViewer(scenario())
    // The decision pane must hide when it loses `.active`. A bare
    // `#pane-decision { ... display ... }` rule (id selector, specificity 100)
    // would override `.pane { display: none }` (specificity 10) and keep the
    // spine permanently visible — breaking the 决策视图/原始时间线 toggle.
    // Only `#pane-decision.active` may set its display.
    expect(html).not.toMatch(/#pane-decision\s*\{[^}]*display/)
    expect(html).toMatch(/#pane-decision\.active\s*\{[^}]*display:\s*flex/)
  })

  it('renders the output answer as markdown (not escaped literal)', () => {
    const events: Event[] = [
      e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1, payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 'llm1', runId: 'r1', type: 'llm.requested', timestamp: 2, causedBy: 'start', payload: { model: 'm' } }),
      e({ id: 'lr1', runId: 'r1', type: 'llm.responded', timestamp: 3, causedBy: 'llm1', payload: {} }),
      e({ id: 'done', runId: 'r1', type: 'agent.run.completed', timestamp: 4, causedBy: 'lr1', payload: { status: 'completed', lastTextOutput: '## 标题\n**重点**' } }),
    ]
    const html = renderViewer(events)
    expect(html).toContain('<h4>标题</h4>')
    expect(html).toContain('<strong>重点</strong>')
  })

  it('trims a panel causal chain to spine decisions only', () => {
    const html = renderViewer(scenario())
    const exps = JSON.parse(html.match(/id="explanations-data">(.*?)<\/script>/s)![1]!)
    const fsmChain = exps['fsm'].chain.map((c: { eventId: string }) => c.eventId)
    expect(fsmChain).not.toContain('lr1')   // llm.responded — non-decision
    expect(fsmChain).not.toContain('tres')  // tool.responded — non-decision
    for (const id of fsmChain) expect(['llm1', 'treq', 'fsm', 'done', 'start']).toContain(id)
  })

  it('shows an honest fallback when the output node has no upstream decision', () => {
    const events: Event[] = [
      e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1, payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 'done', runId: 'r1', type: 'agent.run.completed', timestamp: 2, payload: { status: 'completed', lastTextOutput: 'ok' } }),
    ]
    const html = renderViewer(events)
    const exps = JSON.parse(html.match(/id="explanations-data">(.*?)<\/script>/s)![1]!)
    expect(exps['done'].bodyHtml).toContain('无上游决策记录')
    expect(exps['done'].bodyHtml).not.toContain('点 ← 谁导致的')
  })

  it('renders without crashing for a run with no decisions', () => {
    const html = renderViewer([{ id: 's', runId: 'r1', actor: 'a', type: 'agent.run.started', timestamp: 1, payload: {} } as Event])
    expect(html.startsWith('<!doctype html>')).toBe(true)
    expect(html).toContain('id="why-panel"')
  })
})
