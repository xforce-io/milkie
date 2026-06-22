import { resolveClaimSources } from '../trace/diagnostics/buildLineageProjection'
import type { Event } from '../trace/types'

function objCreated(objectId: string, type: string, meta?: Record<string, unknown>): Event {
  return { id: `${objectId}-c`, runId: 'r', type: 'object.created', actor: 'test', timestamp: 0,
    payload: { objectId, type, producerEventId: 'p', ...(meta ? { meta } : {}) } } as Event
}
function citesRel(from: string, to: string): Event {
  return { id: `${from}-${to}-rel`, runId: 'r', type: 'relation.created', actor: 'test', timestamp: 0,
    payload: { relationId: `${from}-${to}`, type: 'cites', fromObjectId: from, toObjectId: to, causedByEventId: 'x' } } as Event
}

describe('#189 resolveClaimSources', () => {
  const events: Event[] = [
    objCreated('src1', 'passage', { file: 'news.json', source: 'reuters' }),
    objCreated('claim1', 'claim', { text: '实时抓取 228 条' }),
    citesRel('claim1', 'src1'),
  ]
  it('matches a claim by query substring and returns its cited sources', () => {
    const r = resolveClaimSources(events, '228')
    expect(r).toHaveLength(1)
    expect(r[0]!.claim).toContain('228')
    expect(r[0]!.sources).toEqual([{ objectId: 'src1', type: 'passage', meta: { file: 'news.json', source: 'reuters' } }])
  })
  it('returns [] for a non-matching query', () => {
    expect(resolveClaimSources(events, '999')).toEqual([])
  })
  it('returns all claims when query omitted', () => {
    expect(resolveClaimSources(events)).toHaveLength(1)
  })
})
