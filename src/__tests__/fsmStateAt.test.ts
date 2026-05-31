import { fsmStateAt } from '../trace/diagnostics/fsmStateAt'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, payload: unknown): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload })
const trans = (id: string, from: string, to: string) =>
  ev(id, 'fsm.transition', { from, to, trigger: { domain: 'business', name: 'E' } })

describe('fsmStateAt', () => {
  it('returns the last transition target before the event', () => {
    const events: Event[] = [
      trans('t1', 's0', 's1'),
      ev('llm', 'llm.requested', {}),
      trans('t2', 's1', 's2'),
    ]
    expect(fsmStateAt(events, 'llm')).toBe('s1')
  })

  it('returns the initial state (first transition from) when no transition precedes the event', () => {
    const events: Event[] = [
      ev('llm', 'llm.requested', {}),
      trans('t1', 's0', 's1'),
    ]
    expect(fsmStateAt(events, 'llm')).toBe('s0')
  })

  it('returns null when there are no transitions at all', () => {
    const events: Event[] = [ev('llm', 'llm.requested', {})]
    expect(fsmStateAt(events, 'llm')).toBeNull()
  })
})
