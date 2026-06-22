import type { Event, ObjectCreatedPayload, RelationCreatedPayload } from '../types.js'

export interface ClaimLineage {
  claim:   string
  sources: { objectId: string; type: string; meta?: Record<string, unknown> }[]
}

/**
 * #189 ③Lineage read: fold a run's object.created / relation.created into
 * claim→source attributions. Pure (no IO). A claim is an object of type 'claim'
 * whose meta.text holds the conclusion (minted by the `cite` tool); a 'cites'
 * relation points claim→source. `query` filters claims by text substring; omit
 * to return all. Walks the explicit cite graph, NOT event.causedBy.
 */
export function resolveClaimSources(events: Event[], query?: string): ClaimLineage[] {
  const objects = new Map<string, ObjectCreatedPayload>()
  const cites: RelationCreatedPayload[] = []
  for (const e of events) {
    if (e.type === 'object.created') {
      const p = e.payload as ObjectCreatedPayload
      objects.set(p.objectId, p)
    } else if (e.type === 'relation.created') {
      const p = e.payload as RelationCreatedPayload
      if (p.type === 'cites') cites.push(p)
    }
  }
  const out: ClaimLineage[] = []
  for (const [objectId, obj] of objects) {
    if (obj.type !== 'claim') continue
    const text = String(obj.meta?.text ?? '')
    if (query && !text.includes(query)) continue
    const sources = cites
      .filter(c => c.fromObjectId === objectId)
      .map(c => objects.get(c.toObjectId))
      .filter((s): s is ObjectCreatedPayload => !!s)
      .map(s => ({ objectId: s.objectId, type: s.type, ...(s.meta ? { meta: s.meta } : {}) }))
    out.push({ claim: text, sources })
  }
  return out
}
