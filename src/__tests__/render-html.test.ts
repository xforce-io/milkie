import { renderHtml } from '../trace/render/html'
import type { Event } from '../trace/types'

const e = (over: Partial<Event> & { id: string, runId: string, type: Event['type'] }): Event => ({
  actor: 'runtime', timestamp: 0, payload: {}, ...over,
})

describe('renderHtml', () => {
  it('produces a complete HTML document with doctype and trace-data script', () => {
    const events: Event[] = [
      e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'echo', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 'c', runId: 'r1', type: 'agent.run.completed', timestamp: 9,
          payload: { status: 'completed', lastTextOutput: 'hi' } }),
    ]
    const html = renderHtml(events)
    expect(html.startsWith('<!doctype html>')).toBe(true)
    expect(html).toContain('</html>')
    expect(html).toContain('<script type="application/json" id="trace-data">')
    // Embedded JSON must be valid and contain the events.
    const m = html.match(/<script type="application\/json" id="trace-data">([\s\S]*?)<\/script>/)
    expect(m).not.toBeNull()
    const embedded = JSON.parse(m![1]!) as Event[]
    expect(embedded).toHaveLength(2)
    expect(embedded[0]!.id).toBe('s')
  })

  it('renders one timeline section per root run, shows runId in header', () => {
    const events: Event[] = [
      e({ id: 's1', runId: 'parent', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'p', goal: 'g', input: 'i', contextId: 'parent' } }),
      e({ id: 's2', runId: 'child',  type: 'agent.run.started', timestamp: 2,
          payload: { agentId: 'c', goal: 'g', input: 'i', contextId: 'child', parentId: 'parent' } }),
    ]
    const html = renderHtml(events)
    expect(html).toContain('parent')
    expect(html).toContain('child')
    // child should appear nested inside parent's section — assert child's
    // marker comes after parent's section start.
    const parentIdx = html.indexOf('data-run-id="parent"')
    const childIdx  = html.indexOf('data-run-id="child"')
    expect(parentIdx).toBeGreaterThan(-1)
    expect(childIdx).toBeGreaterThan(parentIdx)
  })

  it('escapes HTML-special characters in user payload (XSS guard)', () => {
    const events: Event[] = [
      e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: '<script>alert(1)</script>', goal: 'g', input: 'i', contextId: 'c' } }),
    ]
    const html = renderHtml(events)
    expect(html).not.toContain('<script>alert(1)</script>')
    expect(html).toContain('&lt;script&gt;alert(1)&lt;/script&gt;')
  })

  it('returns valid HTML for an empty event array', () => {
    const html = renderHtml([])
    expect(html.startsWith('<!doctype html>')).toBe(true)
    expect(html).toContain('</html>')
  })

  it('renders the original event payload inside each entry for click-to-expand', () => {
    const events: Event[] = [
      e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'echo', goal: 'g', input: 'i', contextId: 'r1' } }),
      e({ id: 'q', runId: 'r1', type: 'llm.requested', timestamp: 2,
          payload: { request: { messages: [{ role: 'user', content: 'hello-payload-marker' }] }, requestHash: 'h-marker' } }),
      e({ id: 'a', runId: 'r1', type: 'llm.responded', timestamp: 3, causedBy: 'q',
          payload: { response: { content: [{ type: 'text', text: 'response-marker' }] }, requestHash: 'h-marker' } }),
    ]
    const html = renderHtml(events)
    // payload <pre> exists inside an entry
    expect(html).toMatch(/<pre class="payload">[\s\S]*?<\/pre>/)
    // both request and response payload content reachable in the rendered output
    expect(html).toContain('hello-payload-marker')
    expect(html).toContain('response-marker')
  })

  it('whitelists badge class and escapes status text to block attribute-breakout XSS', () => {
    const malicious = '" onerror="alert(1)" x="'
    const events: Event[] = [
      e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'echo', goal: 'g', input: 'i', contextId: 'r1' } }),
      e({ id: 'c', runId: 'r1', type: 'agent.run.completed', timestamp: 9,
          payload: { status: malicious } }),
    ]
    const html = renderHtml(events)
    // attribute breakout must not happen — no raw onerror= reaches the document
    expect(html).not.toContain('onerror="alert(1)"')
    // the status string when rendered as text content is escaped
    expect(html).not.toContain('class="badge ' + malicious)
  })

  it('renders fsm.transition with guard summary in html', () => {
    const events: Event[] = [
      e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 't', runId: 'r1', type: 'fsm.transition', timestamp: 2,
          payload: { from: 'classify', to: 'handle_b', trigger: { domain: 'business', name: 'INTENT_B' },
            guardEvaluations: [{ guardId: 'intent-threshold', result: 'INTENT_B', contextSlice: {} }] } }),
    ]
    const html = renderHtml(events)
    expect(html).toContain('intent-threshold')
    expect(html).toContain('data-kind="fsm"')
  })

  it('renders a Why? block on fsm.transition with anchor links to upstream events', () => {
    const events: Event[] = [
      e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 'treq', runId: 'r1', type: 'tool.requested', timestamp: 2, causedBy: 'start',
          payload: { toolName: 'classify_intent', input: {}, requestHash: 'h' } }),
      e({ id: 'tres', runId: 'r1', type: 'tool.responded', timestamp: 3, causedBy: 'treq',
          payload: { toolName: 'classify_intent', output: {}, requestHash: 'h' } }),
      e({ id: 'fsm', runId: 'r1', type: 'fsm.transition', timestamp: 4, causedBy: 'tres',
          payload: { from: 'classify', to: 'handle_b', trigger: { domain: 'business', name: 'INTENT_B' },
            guardEvaluations: [{ guardId: 'intent-threshold', result: 'INTENT_B', contextSlice: { confidence: 0.9 } }] } }),
    ]
    const html = renderHtml(events)
    expect(html).toContain('class="why"')
    expect(html).toContain('classify → handle_b')
    expect(html).toContain('intent-threshold')
    expect(html).toContain('href="#ev-tres"')
    expect(html).toContain('id="ev-tres"')
  })

  it('renders a Why? block for an fsm.transition without guards', () => {
    const events: Event[] = [
      e({ id: 'start', runId: 'r1', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
      e({ id: 'fsm', runId: 'r1', type: 'fsm.transition', timestamp: 2, causedBy: 'start',
          payload: { from: 's0', to: 'end', trigger: { domain: 'lifecycle', name: 'DONE' } } }),
    ]
    const html = renderHtml(events)
    expect(html).toContain('class="why"')
    expect(html).toContain('s0 → end')
  })

  it('includes Why-block styling', () => {
    const html = renderHtml([
      e({ id: 's', runId: 'r1', type: 'agent.run.started', timestamp: 1,
          payload: { agentId: 'x', goal: 'g', input: 'i', contextId: 'c' } }),
    ])
    expect(html).toContain('.why {')
  })
})

describe('#26 Assembled by', () => {
  const region = (id: string, stability: string, contentHash?: string) =>
    e({ id: `add-${id}`, runId: 'r1', type: 'region.added', timestamp: 1,
        payload: { id, target: 'message', section: 'history', stability, reason: 'turn-archived',
          ...(contentHash ? { contentHash } : {}) } })

  it('renders an Assembled by block on llm.requested with metadata + stability class', () => {
    const events: Event[] = [
      region('header', 'immutable', 'H1'),
      e({ id: 'llm', runId: 'r1', type: 'llm.requested', timestamp: 2, payload: { model: 'm' } }),
    ]
    const html = renderHtml(events, { regionContent: new Map([['H1', 'SYSTEM PROMPT TEXT']]) })
    expect(html).toContain('Assembled by')
    expect(html).toContain('header')
    expect(html).toContain('stab-immutable')
    expect(html).toContain('data-hash="H1"')
    expect(html).toContain('SYSTEM PROMPT TEXT')
  })

  it('dedups identical content across prompts and annotates reuse count', () => {
    const events: Event[] = [
      region('header', 'immutable', 'H1'),
      e({ id: 'llm1', runId: 'r1', type: 'llm.requested', timestamp: 2, payload: { model: 'm' } }),
      e({ id: 'llm2', runId: 'r1', type: 'llm.requested', timestamp: 3, payload: { model: 'm' } }),
    ]
    const html = renderHtml(events, { regionContent: new Map([['H1', 'SHARED-CONTENT-XYZ']]) })
    expect(html.split('SHARED-CONTENT-XYZ').length - 1).toBe(1)
    expect(html).toContain('复用 ×2')
  })

  it('degrades gracefully without region content (metadata only)', () => {
    const events: Event[] = [
      region('header', 'immutable', 'H1'),
      e({ id: 'llm', runId: 'r1', type: 'llm.requested', timestamp: 2, payload: { model: 'm' } }),
    ]
    const html = renderHtml(events)
    expect(html).toContain('Assembled by')
    expect(html).toContain('header')
    expect(html).toContain('(内容不可用)')
  })

  it('renders no Assembled by block for an llm.requested with no active regions', () => {
    const events: Event[] = [
      e({ id: 'llm', runId: 'r1', type: 'llm.requested', timestamp: 2, payload: { model: 'm' } }),
    ]
    const html = renderHtml(events)
    expect(html).not.toContain('Assembled by')
  })

  it('renders a region row without contentHash as metadata only (no data-hash, no preview)', () => {
    const events: Event[] = [
      region('eph', 'volatile'),   // no contentHash
      e({ id: 'llm', runId: 'r1', type: 'llm.requested', timestamp: 2, payload: { model: 'm' } }),
    ]
    const html = renderHtml(events)
    expect(html).toContain('Assembled by')
    expect(html).toContain('eph')
    // no data-hash attribute on the row and no region-preview element emitted
    // (the bare substrings appear in STYLES/SCRIPT, so assert the rendered forms)
    expect(html).not.toContain('data-hash="')
    expect(html).not.toContain('class="region-preview"')
  })

  it('applies the stability class for each stability tier', () => {
    const events: Event[] = [
      region('a', 'immutable', 'HA'),
      region('b', 'session-stable', 'HB'),
      region('c', 'turn-stable', 'HC'),
      region('d', 'volatile', 'HD'),
      e({ id: 'llm', runId: 'r1', type: 'llm.requested', timestamp: 2, payload: { model: 'm' } }),
    ]
    const html = renderHtml(events)
    for (const cls of ['stab-immutable', 'stab-session-stable', 'stab-turn-stable', 'stab-volatile']) {
      expect(html).toContain(cls)
    }
  })

  it('keeps region content with a </script> sequence from breaking the document', () => {
    const events: Event[] = [
      region('x', 'volatile', 'HX'),
      e({ id: 'llm', runId: 'r1', type: 'llm.requested', timestamp: 2, payload: { model: 'm' } }),
    ]
    const html = renderHtml(events, { regionContent: new Map([['HX', '</script><b>boom</b>']]) })
    // the raw closing tag must be neutralized in the embedded JSON registry
    expect(html).not.toContain('</script><b>boom</b>')
    expect(html).toContain('<\\/script>')   // close-tag-safe escaped form present
  })
})
