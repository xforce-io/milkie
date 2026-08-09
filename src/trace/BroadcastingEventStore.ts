import type { ICrashSafeEventStore, IEventStore } from './EventStore.js'
import { isCrashSafeEventStore } from './EventStore.js'
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
  private readonly crashSafeInner: ICrashSafeEventStore | null

  constructor(private readonly inner: IEventStore) {
    this.crashSafeInner = isCrashSafeEventStore(inner) ? inner : null
  }

  /** Present only when the inner store is crash-safe — enables finalization durability. */
  get durability(): 'crash-safe' | undefined {
    return this.crashSafeInner ? 'crash-safe' : undefined
  }

  async append(event: Event): Promise<void> {
    await this.inner.append(event)

    if (event.type === 'agent.run.started') {
      const payload = event.payload as { contextId: string }
      this.contextIdByRunId.set(event.runId, payload.contextId)
    }

    const contextId = await this.contextIdFor(event.runId)
    if (contextId) {
      const subs = this.subscribers.get(contextId)
      if (subs) for (const cb of subs) cb(event)
    }
  }

  /**
   * The contextId owning a run. Normally learned live from the run's
   * `agent.run.started` (cached). But `resume()` reuses a runId without emitting
   * a fresh started event, so a broadcaster created after a serve restart never
   * sees it live — recover the mapping from the persisted log (the started event
   * is durable, e.g. in the JsonlEventStore) so the resumed run's trace/progress
   * events still reach contextId subscribers. Cached so the lookup runs at most
   * once per unknown run.
   */
  private async contextIdFor(runId: string): Promise<string | undefined> {
    const cached = this.contextIdByRunId.get(runId)
    if (cached !== undefined) return cached
    const prior = await this.inner.readByRunId(runId)
    const started = prior.find(e => e.type === 'agent.run.started')
    if (!started) return undefined
    const contextId = (started.payload as { contextId: string }).contextId
    this.contextIdByRunId.set(runId, contextId)
    return contextId
  }

  async readByRunId(runId: string): Promise<Event[]> {
    return this.inner.readByRunId(runId)
  }

  async readRange(runId: string, fromIndex: number, count?: number): Promise<Event[]> {
    return this.inner.readRange(runId, fromIndex, count)
  }

  /**
   * #227: proxy run durability confirmation when the inner store supports it.
   * Throws if the inner store is not crash-safe.
   */
  async confirmRunDurable(runId: string): Promise<void> {
    if (!this.crashSafeInner) {
      throw new Error(
        'BroadcastingEventStore.confirmRunDurable requires a crash-safe inner event store',
      )
    }
    await this.crashSafeInner.confirmRunDurable(runId)
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
