import type { Event, ObjectCreatedPayload, RelationCreatedPayload } from '../types.js'

export interface ClaimLineage {
  claim:   string
  sources: { objectId: string; type: string; meta?: Record<string, unknown> }[]
}

/** #200 A: a content-addressable handle dereferenced to its cite neighbourhood. */
export interface HandleRef {
  objectId: string
  type:     string
  meta?:    Record<string, unknown>
  /** sources this object cites (outgoing `cites`; populated for a claim). */
  cites:    { objectId: string; type: string; meta?: Record<string, unknown> }[]
  /** claims that cite this object (incoming `cites`; populated for a source). */
  citedBy:  { objectId: string; type: string; meta?: Record<string, unknown> }[]
}

/**
 * #200 A: exact, deterministic handle dereference — NO text match. Given an
 * `objectId` (a `claimId` is just a claim object's id), return that object plus its
 * cite neighbourhood: the sources it `cites` (outgoing) and the claims that cite it
 * (`citedBy`, incoming). This is the clean lineage primitive — unlike
 * resolveClaimSources's substring `query` (a weak window-local retrieval), it needs
 * no recall of the verbatim claim text. Returns undefined for an unknown objectId
 * (no fabrication). Pure (no IO); walks the explicit cite graph, NOT event.causedBy.
 */
export function derefObject(events: Event[], objectId: string): HandleRef | undefined {
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
  const self = objects.get(objectId)
  if (!self) return undefined
  const view = (id: string) => {
    const o = objects.get(id)
    return o ? { objectId: o.objectId, type: o.type, ...(o.meta ? { meta: o.meta } : {}) } : undefined
  }
  const isView = <T>(v: T | undefined): v is T => !!v
  return {
    objectId: self.objectId,
    type:     self.type,
    ...(self.meta ? { meta: self.meta } : {}),
    cites:    cites.filter(c => c.fromObjectId === objectId).map(c => view(c.toObjectId)).filter(isView),
    citedBy:  cites.filter(c => c.toObjectId === objectId).map(c => view(c.fromObjectId)).filter(isView),
  }
}

/**
 * #189 ③Lineage read: fold a run's object.created / relation.created into
 * claim→source attributions. Pure (no IO). A claim is an object of type 'claim'
 * whose meta.text holds the conclusion (minted by the `cite` tool); a 'cites'
 * relation points claim→source. `query` is a WEAK window-local retrieval — a
 * case-sensitive substring over claim.meta.text (omit = all). It is NOT the precise
 * lineage entry: for that use derefObject (#200 B keeps these roles distinct).
 * Walks the explicit cite graph, NOT event.causedBy.
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
