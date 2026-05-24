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
})
