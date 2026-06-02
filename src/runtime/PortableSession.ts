import type { Event, AgentSpawnedPayload } from '../trace/types.js'
import type { IEventStore } from '../trace/EventStore.js'
import type { JSONValue } from '../types/common.js'

/**
 * #84: a portable, serialisable snapshot of a session (one contextId), suitable
 * for alfred's persistence layer to store and later re-inject into a fresh Milkie.
 *
 * The event log is the single source of truth (#73): the snapshot bundles the
 * run-tree's events (the latest checkpointed run plus its sub-agent descendants),
 * the context's persistent vars (#83, which live in the stateStore — not the
 * event log — so they are captured separately), and a versioned manifest.
 *
 * Multi-turn note: a context's "history" is carried forward into the latest run's
 * checkpoint (regions/working-memory), so the latest run + its descendants is
 * sufficient to continue the conversation. Prior turns' raw I/O events are not
 * bundled (no by-context index exists, and they are not needed for continuation).
 */
export interface PortableSession {
  manifest: {
    schemaVersion: 1
    contextId:     string
    agentId:       string
    latestRunId:   string
    exportedAt:    number
  }
  events:    Event[]
  variables: Record<string, JSONValue>
}

/** The schema version this build emits and is able to import. */
export const PORTABLE_SESSION_SCHEMA_VERSION = 1

/**
 * Collect a run and all of its sub-agent descendants, in breadth-first order.
 * Descendants are discovered from `agent.spawned` events (each carries the
 * child's independent runId). Cycles/duplicates are guarded by a visited set.
 */
export async function collectRunTree(
  eventStore: IEventStore,
  rootRunId:  string,
): Promise<Event[]> {
  const visited = new Set<string>()
  const queue: string[] = [rootRunId]
  const events: Event[] = []
  while (queue.length > 0) {
    const runId = queue.shift()!
    if (visited.has(runId)) continue
    visited.add(runId)
    const runEvents = await eventStore.readByRunId(runId)
    for (const e of runEvents) {
      events.push(e)
      if (e.type === 'agent.spawned') {
        const childRunId = (e.payload as AgentSpawnedPayload).childRunId
        if (childRunId && !visited.has(childRunId)) queue.push(childRunId)
      }
    }
  }
  return events
}
