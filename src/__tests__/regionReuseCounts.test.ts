import { regionReuseCounts } from '../trace/RegionContextView'
import type { Event } from '../trace/types'

const ev = (id: string, type: string, payload: unknown): Event =>
  ({ id, runId: 'r1', actor: 'a', type: type as Event['type'], timestamp: 0, payload })

const regionAdded = (id: string, contentHash?: string) =>
  ev(`add-${id}`, 'region.added', { id, target: 'message', section: 's', stability: 'volatile', reason: 'r', ...(contentHash ? { contentHash } : {}) })

describe('regionReuseCounts', () => {
  it('counts how many llm.requested active-sets reference each contentHash', () => {
    const events: Event[] = [
      regionAdded('h', 'HASH'),
      ev('llm1', 'llm.requested', {}),
      ev('llm2', 'llm.requested', {}),
    ]
    const counts = regionReuseCounts(events)
    expect(counts.get('HASH')).toBe(2)
  })

  it('single reference counts as 1', () => {
    const events: Event[] = [regionAdded('h', 'HASH'), ev('llm1', 'llm.requested', {})]
    expect(regionReuseCounts(events).get('HASH')).toBe(1)
  })

  it('ignores regions without a contentHash and llm calls with no active regions', () => {
    const events: Event[] = [
      ev('llm0', 'llm.requested', {}),
      regionAdded('h'),
      ev('llm1', 'llm.requested', {}),
    ]
    expect(regionReuseCounts(events).size).toBe(0)
  })
})
