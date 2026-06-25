import { resolveClaimSources, derefObject } from '../trace/diagnostics/buildLineageProjection'
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

describe('#200 A: derefObject — exact handle dereference (no text match)', () => {
  const events: Event[] = [
    objCreated('src1', 'shell:stdout', { source: 'cnbc', url: 'https://x' }),
    objCreated('claim1', 'claim', { text: '某信号 — 原文地址' }),
    citesRel('claim1', 'src1'),
  ]

  it('dereferences a claim by its objectId → its type/meta and the sources it cites', () => {
    const r = derefObject(events, 'claim1')!
    expect(r.objectId).toBe('claim1')
    expect(r.type).toBe('claim')
    expect(r.meta).toEqual({ text: '某信号 — 原文地址' })
    expect(r.cites).toEqual([{ objectId: 'src1', type: 'shell:stdout', meta: { source: 'cnbc', url: 'https://x' } }])
    expect(r.citedBy).toEqual([])
  })

  it('dereferences a source object by its objectId → the claims that cite it (reverse)', () => {
    const r = derefObject(events, 'src1')!
    expect(r.objectId).toBe('src1')
    expect(r.type).toBe('shell:stdout')
    expect(r.cites).toEqual([])
    expect(r.citedBy).toEqual([{ objectId: 'claim1', type: 'claim', meta: { text: '某信号 — 原文地址' } }])
  })

  it('returns undefined for an unknown objectId (no fabrication)', () => {
    expect(derefObject(events, 'nope')).toBeUndefined()
  })
})
