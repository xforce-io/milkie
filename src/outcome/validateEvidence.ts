/**
 * Evidence validation for task outcome finalization (#227 / s-017).
 * Verifies run lifecycle, event/object uniqueness, content hashes, and durability.
 */

import type { Event, ObjectCreatedPayload } from '../trace/types.js'
import type { IEventStore } from '../trace/EventStore.js'
import {
  isCrashSafeEventStore,
  type ICrashSafeEventStore,
} from '../trace/EventStore.js'
import type { ITraceObjectStore } from '../trace/TraceObjectStore.js'
import {
  isCrashSafeTraceObjectStore,
  type ICrashSafeTraceObjectStore,
} from '../trace/TraceObjectStore.js'
import { contentAddressForCanonicalBytes } from '../trace/hash.js'
import type {
  DurabilityClass,
  EvidenceRef,
  TaskOutcomeFinalization,
} from '../types/outcome.js'
import {
  TaskOutcomeEvidenceError,
  TaskOutcomeFinalizationConfigurationError,
} from '../types/outcome.js'

export interface EvidenceValidationContext {
  readonly events: readonly Event[]
  readonly eventById: ReadonlyMap<string, Event>
  readonly completedCount: number
}

/**
 * Build event id map and enforce uniqueness + non-empty ids.
 * Exactly one `agent.run.completed` is required for finalization.
 */
export function buildEvidenceContext(events: readonly Event[]): EvidenceValidationContext {
  const eventById = new Map<string, Event>()
  let completedCount = 0

  for (const ev of events) {
    if (typeof ev.id !== 'string' || ev.id.length === 0) {
      throw new TaskOutcomeEvidenceError(
        'empty_event_id',
        'run contains an event with empty id',
      )
    }
    if (eventById.has(ev.id)) {
      throw new TaskOutcomeEvidenceError(
        'duplicate_event_id',
        'run contains duplicate event ids',
      )
    }
    eventById.set(ev.id, ev)
    if (ev.type === 'agent.run.completed') completedCount++
  }

  if (completedCount !== 1) {
    throw new TaskOutcomeEvidenceError(
      completedCount === 0 ? 'run_not_completed' : 'completed_count_invalid',
      completedCount === 0
        ? 'run has no agent.run.completed event'
        : `run has ${completedCount} agent.run.completed events; exactly 1 required`,
    )
  }

  return { events, eventById, completedCount }
}

function validateEventEvidence(
  ref: Extract<EvidenceRef, { kind: 'event' }>,
  ctx: EvidenceValidationContext,
): void {
  if (!ctx.eventById.has(ref.eventId)) {
    throw new TaskOutcomeEvidenceError(
      'event_not_found',
      'event evidence not found in run',
    )
  }
}

async function validateObjectEvidence(
  ref: Extract<EvidenceRef, { kind: 'object' }>,
  ctx: EvidenceValidationContext,
  objectStore: ITraceObjectStore | null | undefined,
): Promise<void> {
  if (!objectStore) {
    throw new TaskOutcomeEvidenceError(
      'object_store_required',
      'object evidence requires traceObjectStore',
    )
  }

  const matches: ObjectCreatedPayload[] = []
  for (const ev of ctx.events) {
    if (ev.type !== 'object.created') continue
    const payload = ev.payload
    if (
      payload !== null &&
      typeof payload === 'object' &&
      'objectId' in payload &&
      (payload as ObjectCreatedPayload).objectId === ref.objectId
    ) {
      matches.push(payload as ObjectCreatedPayload)
    }
  }

  if (matches.length === 0) {
    throw new TaskOutcomeEvidenceError(
      'object_not_found',
      'object evidence object.created not found in run',
    )
  }
  if (matches.length > 1) {
    throw new TaskOutcomeEvidenceError(
      'object_ambiguous',
      'object evidence matches multiple object.created events',
    )
  }

  const created = matches[0]!
  if (typeof created.hash !== 'string' || created.hash.length === 0) {
    throw new TaskOutcomeEvidenceError(
      'object_hash_missing',
      'object.created event is missing hash; legacy objects cannot be finalized',
    )
  }
  if (created.hash !== ref.hash) {
    throw new TaskOutcomeEvidenceError(
      'object_hash_mismatch',
      'object evidence hash does not match object.created payload.hash',
    )
  }

  let bytes: string | undefined
  try {
    bytes = await objectStore.getCanonical(ref.hash)
  } catch (err) {
    throw new TaskOutcomeEvidenceError(
      'object_bytes_missing',
      'object store failed to load canonical bytes',
    )
  }
  if (bytes === undefined) {
    throw new TaskOutcomeEvidenceError(
      'object_bytes_missing',
      'object store has no canonical bytes for evidence hash',
    )
  }

  const recomputed = contentAddressForCanonicalBytes(bytes)
  if (recomputed !== ref.hash || recomputed !== created.hash) {
    throw new TaskOutcomeEvidenceError(
      'object_bytes_hash_mismatch',
      'object canonical bytes hash does not match evidence/event hash',
    )
  }
}

/**
 * Validate every evidence ref against the run. Never calls final store create.
 */
export async function validateEvidenceRefs(
  evidence: readonly EvidenceRef[],
  ctx: EvidenceValidationContext,
  objectStore: ITraceObjectStore | null | undefined,
): Promise<void> {
  for (const ref of evidence) {
    if (ref.kind === 'event') {
      validateEventEvidence(ref, ctx)
    } else {
      await validateObjectEvidence(ref, ctx, objectStore)
    }
  }
}

export function assertDurabilityCompatibility(args: {
  finalDurability: DurabilityClass
  eventStore: IEventStore
  objectStore: ITraceObjectStore | null | undefined
  hasObjectEvidence: boolean
}): { crashSafeEvent?: ICrashSafeEventStore; crashSafeObject?: ICrashSafeTraceObjectStore } {
  const { finalDurability, eventStore, objectStore, hasObjectEvidence } = args

  if (finalDurability === 'crash-safe') {
    if (!isCrashSafeEventStore(eventStore)) {
      throw new TaskOutcomeFinalizationConfigurationError(
        'crash-safe finalization requires an eventStore implementing ICrashSafeEventStore',
      )
    }
    if (hasObjectEvidence) {
      if (!objectStore || !isCrashSafeTraceObjectStore(objectStore)) {
        throw new TaskOutcomeFinalizationConfigurationError(
          'crash-safe finalization with object evidence requires a traceObjectStore implementing ICrashSafeTraceObjectStore',
        )
      }
      return { crashSafeEvent: eventStore, crashSafeObject: objectStore }
    }
    return { crashSafeEvent: eventStore }
  }

  // process final: evidence stores need not be crash-safe; result is process-lifetime only.
  return {}
}

/**
 * Confirm evidence durability before final create (crash-safe path only).
 * Order: run events first, then object bytes/directories.
 */
export async function confirmEvidenceDurable(args: {
  runId: string
  evidence: readonly EvidenceRef[]
  crashSafeEvent?: ICrashSafeEventStore
  crashSafeObject?: ICrashSafeTraceObjectStore
}): Promise<void> {
  const { runId, evidence, crashSafeEvent, crashSafeObject } = args
  if (!crashSafeEvent) return

  try {
    await crashSafeEvent.confirmRunDurable(runId)
  } catch (err) {
    throw new TaskOutcomeEvidenceError(
      'durability_confirm_failed',
      'failed to confirm run event durability before finalization',
    )
  }

  if (!crashSafeObject) return

  const hashes: `sha256:${string}`[] = []
  const seen = new Set<string>()
  for (const ref of evidence) {
    if (ref.kind !== 'object') continue
    if (seen.has(ref.hash)) continue
    seen.add(ref.hash)
    hashes.push(ref.hash)
  }
  if (hashes.length === 0) return

  try {
    await crashSafeObject.confirmObjectsDurable(hashes)
  } catch (err) {
    throw new TaskOutcomeEvidenceError(
      'durability_confirm_failed',
      'failed to confirm object durability before finalization',
    )
  }
}

/**
 * Resolve attempt result against an existing finalization (idempotency table).
 */
export function resolveAgainstExisting(
  existing: TaskOutcomeFinalization,
  attempted: {
    finalizationId: string
    value: TaskOutcomeFinalization['value']
    intentHash: TaskOutcomeFinalization['intentHash']
  },
):
  | { readonly status: 'idempotent'; readonly final: TaskOutcomeFinalization }
  | {
      readonly status: 'conflict'
      readonly existing: TaskOutcomeFinalization
      readonly conflict: {
        readonly kind: 'already_finalized' | 'idempotency_key_reused'
        readonly attempted: {
          readonly finalizationId: string
          readonly value: TaskOutcomeFinalization['value']
          readonly intentHash: TaskOutcomeFinalization['intentHash']
        }
      }
    } {
  if (existing.finalizationId === attempted.finalizationId) {
    if (existing.intentHash === attempted.intentHash) {
      return { status: 'idempotent', final: existing }
    }
    return {
      status: 'conflict',
      existing,
      conflict: {
        kind: 'idempotency_key_reused',
        attempted: {
          finalizationId: attempted.finalizationId,
          value: attempted.value,
          intentHash: attempted.intentHash,
        },
      },
    }
  }
  return {
    status: 'conflict',
    existing,
    conflict: {
      kind: 'already_finalized',
      attempted: {
        finalizationId: attempted.finalizationId,
        value: attempted.value,
        intentHash: attempted.intentHash,
      },
    },
  }
}
