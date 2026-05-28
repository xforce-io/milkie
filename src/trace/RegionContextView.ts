import type { Event, RegionAddedPayload, RegionRemovedPayload } from './types.js'
import type { ITraceObjectStore } from './TraceObjectStore.js'

export interface RegionContentRef {
  id:           string
  target:       RegionAddedPayload['target']
  section:      string
  stability:    RegionAddedPayload['stability']
  reason:       string
  contentHash:  string
  renderedHash?: string
  content?:     string
  rendered?:    string
}

export type ContextFoldMode = 'at' | 'before'

function eventsThrough(events: Event[], eventId: string, mode: ContextFoldMode): Event[] {
  const index = events.findIndex(e => e.id === eventId)
  if (index < 0) throw new Error(`Event not found: ${eventId}`)
  return events.slice(0, mode === 'at' ? index + 1 : index)
}

export function foldRegionContext(events: Event[]): Map<string, RegionContentRef> {
  const active = new Map<string, RegionContentRef>()
  for (const event of events) {
    if (event.type === 'region.added') {
      const payload = event.payload as RegionAddedPayload
      active.set(payload.id, {
        id:           payload.id,
        target:       payload.target,
        section:      payload.section,
        stability:    payload.stability,
        reason:       payload.reason,
        contentHash:  payload.contentHash,
        ...(payload.renderedHash ? { renderedHash: payload.renderedHash } : {}),
      })
    } else if (event.type === 'region.removed') {
      const payload = event.payload as RegionRemovedPayload
      active.delete(payload.id)
    }
  }
  return active
}

export function contextRefsAt(events: Event[], eventId: string, mode: ContextFoldMode = 'at'): Map<string, RegionContentRef> {
  return foldRegionContext(eventsThrough(events, eventId, mode))
}

export async function hydrateRegionContext(
  refs: Map<string, RegionContentRef>,
  objectStore: ITraceObjectStore,
): Promise<Map<string, RegionContentRef>> {
  const hydrated = new Map<string, RegionContentRef>()
  for (const [id, ref] of refs) {
    const rendered = ref.renderedHash ? await objectStore.getCanonical(ref.renderedHash) : undefined
    hydrated.set(id, {
      ...ref,
      content:  await objectStore.getCanonical(ref.contentHash),
      ...(rendered !== undefined ? { rendered } : {}),
    })
  }
  return hydrated
}

export async function contextAt(
  events: Event[],
  eventId: string,
  objectStore: ITraceObjectStore,
): Promise<Map<string, RegionContentRef>> {
  return hydrateRegionContext(contextRefsAt(events, eventId, 'at'), objectStore)
}

export async function contextBefore(
  events: Event[],
  eventId: string,
  objectStore: ITraceObjectStore,
): Promise<Map<string, RegionContentRef>> {
  return hydrateRegionContext(contextRefsAt(events, eventId, 'before'), objectStore)
}

export async function getRegionAt(
  events: Event[],
  eventId: string,
  regionId: string,
  objectStore: ITraceObjectStore,
): Promise<RegionContentRef | undefined> {
  return (await contextAt(events, eventId, objectStore)).get(regionId)
}

export async function getRegionBefore(
  events: Event[],
  eventId: string,
  regionId: string,
  objectStore: ITraceObjectStore,
): Promise<RegionContentRef | undefined> {
  return (await contextBefore(events, eventId, objectStore)).get(regionId)
}
