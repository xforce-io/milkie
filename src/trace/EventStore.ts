import type { Event } from './types.js'

/**
 * Append-only event store interface for Agent Trace.
 *
 * Implementations:
 *   - MemoryEventStore: process-local Map; tests / demos
 *   - JsonlEventStore: file-per-run JSONL; durable
 *
 * The interface is intentionally minimal in Phase 2. Phase 3+ will add
 * indexing (by event type, by hash) for content-addressed cache lookup.
 */
export interface IEventStore {
  /**
   * Append a single event. Implementations must be append-only — the
   * event log is the run's source of truth and cannot be mutated.
   */
  append(event: Event): Promise<void>

  /**
   * Read all events for a given run, in append order.
   * Returns an empty array if the run is unknown.
   */
  readByRunId(runId: string): Promise<Event[]>

  /**
   * Read a window of events for a run, in append order.
   * `fromIndex` is inclusive; if `count` is omitted, returns to end.
   */
  readRange(runId: string, fromIndex: number, count?: number): Promise<Event[]>
}

/**
 * Optional crash-safe durability capability for evidence used by
 * task outcome finalization (#227 / s-017). Memory stores do not implement this.
 */
export interface ICrashSafeEventStore extends IEventStore {
  readonly durability: 'crash-safe'
  /**
   * Confirm the run's event file (and parent directory) are durable on stable
   * storage. Must only be called after the run's events are already readable.
   */
  confirmRunDurable(runId: string): Promise<void>
}

export function isCrashSafeEventStore(store: IEventStore): store is ICrashSafeEventStore {
  const candidate = store as ICrashSafeEventStore
  return (
    candidate !== null &&
    typeof candidate === 'object' &&
    candidate.durability === 'crash-safe' &&
    typeof candidate.confirmRunDurable === 'function'
  )
}
