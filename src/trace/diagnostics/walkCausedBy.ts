import type { Event } from '../types.js'

/**
 * Walk the causedBy chain upstream from `startId`, nearest-first, until an
 * event has no causedBy, its causedBy points outside `events`, or a cycle is
 * detected. Pure: reads only the provided events. Reusable by #32/#36.
 */
export function walkCausedBy(events: Event[], startId: string): Event[] {
  const byId = new Map<string, Event>()
  for (const e of events) byId.set(e.id, e)

  const chain: Event[] = []
  const seen = new Set<string>()
  let current = byId.get(startId)
  while (current && !seen.has(current.id)) {
    seen.add(current.id)
    chain.push(current)
    current = current.causedBy ? byId.get(current.causedBy) : undefined
  }
  return chain
}
