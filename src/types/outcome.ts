/**
 * Task outcome — task-level judgment independent of execution status.
 * See ARCHITECTURE.md Outcome concept and invariant 14; stories s-016 / s-017.
 *
 * Two independent surfaces:
 *   - Observation (s-016): append-only `task.outcome.recorded`, last-write-wins query.
 *   - Finalization (s-017): immutable, evidence-bound final result, create-if-absent once.
 */

export type TaskOutcomeValue = 'success' | 'failure' | 'partial' | 'unknown'

/** Who recorded the judgment (eval harness, human, rule, business callback, …). */
export type TaskOutcomeSource = string

export interface TaskOutcomeScore {
  name:  string
  value: number | string | boolean
}

export interface RecordTaskOutcomeInput {
  runId:   string
  value:   TaskOutcomeValue
  /** Non-empty; recommended: eval | human | rule | business */
  source:  TaskOutcomeSource
  note?:   string
  scores?: TaskOutcomeScore[]
}

/** Query view of the latest outcome recorded for a run. */
export interface TaskOutcome {
  runId:      string
  value:      TaskOutcomeValue
  source:     TaskOutcomeSource
  recordedAt: number
  note?:      string
  scores?:    TaskOutcomeScore[]
}

export class TaskOutcomeRunNotFoundError extends Error {
  readonly runId: string
  constructor(runId: string) {
    super(`No run found for runId "${runId}" (no events in event store)`)
    this.name  = 'TaskOutcomeRunNotFoundError'
    this.runId = runId
  }
}

export class TaskOutcomeError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'TaskOutcomeError'
  }
}

// ─── #227 / s-017: immutable task outcome finalization ───────────────────────

export type VerifierClaimType = 'human' | 'eval' | 'rule' | 'service'

export interface VerifierClaim {
  readonly type: VerifierClaimType
  readonly id: string
}

export type EvidenceRef =
  | { readonly kind: 'event'; readonly eventId: string }
  | { readonly kind: 'object'; readonly objectId: string; readonly hash: `sha256:${string}` }

export interface FinalizeTaskOutcomeInput {
  readonly runId: string
  readonly expectedState: 'unfinalized'
  readonly finalizationId: string
  readonly value: TaskOutcomeValue
  readonly verifierClaim: VerifierClaim
  readonly evidence: readonly EvidenceRef[]
  readonly note?: string
  readonly scores?: readonly TaskOutcomeScore[]
}

export interface TaskOutcomeFinalization {
  readonly schemaVersion: 1
  readonly state: 'finalized'
  readonly runId: string
  readonly value: TaskOutcomeValue
  readonly verifierClaim: VerifierClaim
  readonly evidence: readonly EvidenceRef[]
  readonly note?: string
  readonly scores?: readonly TaskOutcomeScore[]
  readonly finalizationId: string
  readonly intentHash: `sha256:${string}`
  readonly finalizedAt: number
  readonly recordHash: `sha256:${string}`
}

export type FinalizationConflictKind =
  | 'already_finalized'
  | 'idempotency_key_reused'

export type FinalizationAttemptResult =
  | { readonly status: 'finalized'; readonly final: TaskOutcomeFinalization }
  | { readonly status: 'idempotent'; readonly final: TaskOutcomeFinalization }
  | {
      readonly status: 'conflict'
      readonly existing: TaskOutcomeFinalization
      readonly conflict: {
        readonly kind: FinalizationConflictKind
        readonly attempted: {
          readonly finalizationId: string
          readonly value: TaskOutcomeValue
          readonly intentHash: `sha256:${string}`
        }
      }
    }

export type DurabilityClass = 'process' | 'crash-safe'

export type TaskOutcomeEvidenceErrorReason =
  | 'run_not_completed'
  | 'completed_count_invalid'
  | 'duplicate_event_id'
  | 'empty_event_id'
  | 'event_not_found'
  | 'object_not_found'
  | 'object_ambiguous'
  | 'object_hash_missing'
  | 'object_hash_mismatch'
  | 'object_bytes_missing'
  | 'object_bytes_hash_mismatch'
  | 'object_store_required'
  | 'evidence_empty'
  | 'evidence_too_many'
  | 'evidence_duplicate'
  | 'durability_confirm_failed'

export class TaskOutcomeFinalizationValidationError extends Error {
  readonly code = 'task_outcome_finalization_validation' as const
  constructor(message: string) {
    super(message)
    this.name = 'TaskOutcomeFinalizationValidationError'
  }
}

export class TaskOutcomeFinalizationConfigurationError extends Error {
  readonly code = 'task_outcome_finalization_configuration' as const
  constructor(message: string) {
    super(message)
    this.name = 'TaskOutcomeFinalizationConfigurationError'
  }
}

export class TaskOutcomeEvidenceError extends Error {
  readonly code = 'task_outcome_evidence' as const
  readonly reason: TaskOutcomeEvidenceErrorReason
  constructor(reason: TaskOutcomeEvidenceErrorReason, message: string) {
    super(message)
    this.name = 'TaskOutcomeEvidenceError'
    this.reason = reason
  }
}

export type TaskOutcomeFinalizationStoreErrorKind =
  | 'commit_unknown'
  | 'io'
  | 'invalid_record'

export class TaskOutcomeFinalizationStoreError extends Error {
  readonly code = 'task_outcome_finalization_store' as const
  readonly kind: TaskOutcomeFinalizationStoreErrorKind
  readonly stage: string
  readonly cause?: unknown
  constructor(
    kind: TaskOutcomeFinalizationStoreErrorKind,
    stage: string,
    message: string,
    cause?: unknown,
  ) {
    super(message)
    this.name = 'TaskOutcomeFinalizationStoreError'
    this.kind = kind
    this.stage = stage
    this.cause = cause
  }
}

export class TaskOutcomeFinalizationCorruptionError extends Error {
  readonly code = 'task_outcome_finalization_corruption' as const
  readonly runId: string
  constructor(runId: string, message: string) {
    super(message)
    this.name = 'TaskOutcomeFinalizationCorruptionError'
    this.runId = runId
  }
}
