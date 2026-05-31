import type { IEventStore } from './EventStore.js'
import type { Event } from './types.js'

/**
 * Process-local in-memory event store. Loses data on restart. Default
 * for tests and demos; durable workloads should use JsonlEventStore or
 * a backend-specific implementation.
 */
export class MemoryEventStore implements IEventStore {
  private readonly events: Map<string, Event[]> = new Map()

  async append(event: Event): Promise<void> {
    // Store an immutable frozen snapshot, matching durable stores (JsonlEventStore
    // serialises on write). Without this, an in-place mutation of a payload object
    // still referenced by live state (e.g. a plan shared by a tool output and
    // working memory) would retroactively rewrite the recorded event, breaking
    // deterministic replay (the cached value would differ from what was sent).
    let frozen: Event
    try {
      frozen = JSON.parse(JSON.stringify(event)) as Event
    } catch {
      frozen = event  // non-serialisable payload (circular/BigInt/…): cannot freeze; store as-is
    }
    const list = this.events.get(event.runId)
    if (list) {
      list.push(frozen)
    } else {
      this.events.set(event.runId, [frozen])
    }
  }

  async readByRunId(runId: string): Promise<Event[]> {
    return [...(this.events.get(runId) ?? [])]
  }

  async readRange(runId: string, fromIndex: number, count?: number): Promise<Event[]> {
    const all = this.events.get(runId) ?? []
    return count !== undefined
      ? all.slice(fromIndex, fromIndex + count)
      : all.slice(fromIndex)
  }
}
