import { renderViewer } from '../trace/render/viewer'
import type { Event } from '../trace/types'

const e = (over: Partial<Event> & { id: string; runId: string; type: Event['type'] }): Event =>
  ({ actor: 'a', timestamp: 0, payload: {}, ...over })

// #175 de-core: the decision spine is anchored on I/O EFFECTS — `llm.responded`
// and `tool.responded` — plus the final output, not on `fsm.transition` business
// nodes. The spine here is: lr1 (llm decision) → tres (tool result) → done.
function scenario(): Event[] {
  return [
    e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1, payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
    e({ id: 'llm1', runId: 'r1', type: 'llm.requested', timestamp: 2, causedBy: 'start', payload: { model: 'm' } }),
    e({ id: 'lr1', runId: 'r1', type: 'llm.responded', timestamp: 3, causedBy: 'llm1', payload: { response: { content: [], toolCalls: [{ name: 'classify_intent', input: {} }] }, requestHash: 'h1' } }),
    e({ id: 'treq', runId: 'r1', type: 'tool.requested', timestamp: 4, causedBy: 'lr1', payload: { toolName: 'classify_intent', input: {}, requestHash: 'h' } }),
    e({ id: 'tres', runId: 'r1', type: 'tool.responded', timestamp: 5, causedBy: 'treq', payload: { toolName: 'classify_intent', output: {}, status: 'ok', requestHash: 'h' } }),
    e({ id: 'done', runId: 'r1', type: 'agent.run.completed', timestamp: 6, causedBy: 'tres', payload: { status: 'completed', lastTextOutput: 'ok' } }),
  ]
}

describe('renderViewer', () => {
  it('produces a self-contained document with a decision spine and embedded explanations', () => {
    const html = renderViewer(scenario())
    expect(html.startsWith('<!doctype html>')).toBe(true)
    // spine has nodes with data-id for each decision (anchored on effects)
    expect(html).toContain('data-id="lr1"')   // llm.responded — model decision
    expect(html).toContain('data-id="tres"')  // tool.responded — tool result
    expect(html).toContain('data-id="done"')  // agent.run.completed — output
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
      e({ id: 'lr1', runId: 'r1', type: 'llm.responded', timestamp: 3, causedBy: 'llm1', payload: { response: { content: [], toolCalls: [] }, requestHash: 'h' } }),
      e({ id: 'done', runId: 'r1', type: 'agent.run.completed', timestamp: 4, causedBy: 'lr1', payload: { status: 'completed', lastTextOutput: '## 标题\n**重点**' } }),
    ]
    const html = renderViewer(events)
    expect(html).toContain('<h4>标题</h4>')
    expect(html).toContain('<strong>重点</strong>')
  })

  it('renders structured model error metadata on a failed output', () => {
    const events: Event[] = [
      e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1, payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 'done', runId: 'r1', type: 'agent.run.completed', timestamp: 2, payload: {
        status: 'error', lastTextOutput: 'Model provider connection failed.',
        error: {
          code: 'MODEL_CONNECTION_ERROR', message: 'Model provider connection failed.',
          phase: 'stream_open', provider: 'volcengine', model: 'glm-5.2', retryable: true,
        },
      } }),
    ]
    const html = renderViewer(events)
    expect(html).toContain('MODEL_CONNECTION_ERROR')
    expect(html).toContain('stream_open')
    expect(html).toContain('volcengine / glm-5.2')
    expect(html).toContain('retryable')
  })

  it('trims a panel causal chain to spine decisions only', () => {
    const html = renderViewer(scenario())
    const exps = JSON.parse(html.match(/id="explanations-data">(.*?)<\/script>/s)![1]!)
    // tool.responded `tres` is a decision; its raw causedBy chain walks through
    // treq → lr1 → llm1 → start, but the panel chain must keep only spine
    // decision events (lr1, tres) and drop the non-decision intermediates.
    const tresChain = exps['tres'].chain.map((c: { eventId: string }) => c.eventId)
    expect(tresChain).not.toContain('treq')  // tool.requested — non-decision
    expect(tresChain).not.toContain('llm1')  // llm.requested — non-decision
    expect(tresChain).not.toContain('start') // run started — non-decision
    for (const id of tresChain) expect(['lr1', 'tres', 'done']).toContain(id)
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
    const html = renderViewer([{ id: 's', runId: 'r1', actor: 'a', type: 'agent.run.started', timestamp: 1, payload: {} as Event['payload'] } as Event])
    expect(html.startsWith('<!doctype html>')).toBe(true)
    expect(html).toContain('id="why-panel"')
  })

  it('does not leak secrets from a tampered llm.responded terminal in rawJson or timeline', () => {
    const SECRET = 'sk-viewer-tampered-token-LEAKME'
    const events: Event[] = [
      e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 'llm1', runId: 'r1', type: 'llm.requested', timestamp: 2, causedBy: 'start',
          payload: {
            request: { model: 'm', messages: [] },
            requestHash: 'h1',
            outcomeSchemaVersion: 2,
          } }),
      e({ id: 'lr1', runId: 'r1', type: 'llm.responded', timestamp: 3, causedBy: 'llm1',
          payload: {
            status: 'error',
            requestHash: 'h1',
            error: {
              code: 'MODEL_AUTH_ERROR',
              message: 'Model provider authentication failed.',
              phase: 'request',
              provider: 'anthropic',
              model: 'm',
              retryable: false,
              stack: SECRET,
              cause: SECRET,
            },
            token: SECRET,
          } }),
      e({ id: 'done', runId: 'r1', type: 'agent.run.completed', timestamp: 4, causedBy: 'lr1',
          payload: { status: 'error', lastTextOutput: 'failed' } }),
    ]
    const html = renderViewer(events)
    expect(html).not.toContain(SECRET)
    expect(html).toContain('malformed_payload')
    const exps = JSON.parse(html.match(/id="explanations-data">(.*?)<\/script>/s)![1]!)
    expect(exps['lr1'].rawJson).not.toContain(SECRET)
    expect(exps['lr1'].rawJson).toContain('malformed')
    // Raw timeline tab reuses sanitized embed.
    const embedded = html.match(/id="trace-data">(.*?)<\/script>/s)?.[1] ?? ''
    expect(embedded).not.toContain(SECRET)
  })
})
