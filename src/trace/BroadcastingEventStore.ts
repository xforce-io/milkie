import type { IEventStore } from './EventStore.js'
import type { Event } from './types.js'

type Subscriber = (event: Event) => void

/**
 * IEventStore decorator that:
 *  1. Delegates persistence to an inner store (typically MemoryEventStore /
 *     JsonlEventStore).
 *  2. Tracks runId → contextId mappings via `agent.run.started` events
 *     observed during append().
 *  3. Broadcasts each appended event to subscribers registered for the
 *     event's contextId.
 *
 * The runId → contextId cache survives only as long as this instance (i.e.,
 * the server process). Used by `milkie serve` (#86) to fan persistent trace
 * events out to a per-contextId SSE stream, and by examples/agent-docs-qa's
 * web server.
 *
 * Callers MUST invoke the returned unsubscribe function (e.g. on SSE close)
 * to prevent memory leaks.
 */
export class BroadcastingEventStore implements IEventStore {
  private readonly subscribers: Map<string, Set<Subscriber>> = new Map()
  private readonly contextIdByRunId: Map<string, string> = new Map()

  constructor(private readonly inner: IEventStore) {}

  async append(event: Event): Promise<void> {
    await this.inner.append(event)

    if (event.type === 'agent.run.started') {
      const payload = event.payload as { contextId: string }
      this.contextIdByRunId.set(event.runId, payload.contextId)
    }

    const contextId = this.contextIdByRunId.get(event.runId)
    if (contextId) {
      const subs = this.subscribers.get(contextId)
      if (subs) for (const cb of subs) cb(event)
    }
  }

  async readByRunId(runId: string): Promise<Event[]> {
    return this.inner.readByRunId(runId)
  }

  async readRange(runId: string, fromIndex: number, count?: number): Promise<Event[]> {
    return this.inner.readRange(runId, fromIndex, count)
  }

  /**
   * Subscribe to live appended events for a given contextId.
   * Returns an unsubscribe function. Caller must invoke it on stream close.
   */
  subscribe(contextId: string, cb: Subscriber): () => void {
    let set = this.subscribers.get(contextId)
    if (!set) {
      set = new Set()
      this.subscribers.set(contextId, set)
    }
    set.add(cb)
    return () => { set!.delete(cb) }
  }
}
