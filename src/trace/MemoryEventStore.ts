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
    const list = this.events.get(event.runId)
    if (list) {
      list.push(event)
    } else {
      this.events.set(event.runId, [event])
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
