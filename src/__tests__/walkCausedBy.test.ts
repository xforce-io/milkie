import { walkCausedBy } from '../trace/diagnostics/walkCausedBy'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, causedBy?: string): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload: {}, ...(causedBy ? { causedBy } : {}) })

describe('walkCausedBy', () => {
  it('walks the causedBy chain from an event up to the root', () => {
    const events: Event[] = [
      ev('start', 'agent.run.started'),
      ev('llm1', 'llm.responded', 'start'),
      ev('treq', 'tool.requested', 'llm1'),
      ev('tres', 'tool.responded', 'treq'),
      ev('fsm', 'fsm.transition', 'tres'),
    ]
    const chain = walkCausedBy(events, 'fsm')
    expect(chain.map(e => e.id)).toEqual(['fsm', 'tres', 'treq', 'llm1', 'start'])
  })

  it('stops gracefully when a causedBy points to a missing event', () => {
    const events: Event[] = [ev('fsm', 'fsm.transition', 'gone')]
    const chain = walkCausedBy(events, 'fsm')
    expect(chain.map(e => e.id)).toEqual(['fsm'])
  })

  it('does not loop forever on a cycle', () => {
    const events: Event[] = [ev('a', 'x', 'b'), ev('b', 'x', 'a')]
    const chain = walkCausedBy(events, 'a')
    expect(chain.map(e => e.id)).toEqual(['a', 'b'])
  })

  it('returns empty when the start id is unknown', () => {
    expect(walkCausedBy([], 'nope')).toEqual([])
  })
})
