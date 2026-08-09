/**
 * Canonical preimage + hash helpers for task outcome finalization (#227 / s-017).
 * All finalization implementations MUST use these helpers exclusively.
 */

import { canonicalize, hashCanonical } from '../trace/hash.js'
import type {
  EvidenceRef,
  TaskOutcomeFinalization,
  TaskOutcomeScore,
  TaskOutcomeValue,
  VerifierClaim,
} from '../types/outcome.js'
import { TaskOutcomeFinalizationValidationError } from '../types/outcome.js'

const SHA256_RE = /^sha256:[a-f0-9]{64}$/

export type ContentAddress = `sha256:${string}`

export interface NormalizedFinalizationIntent {
  readonly schemaVersion: 1
  readonly runId: string
  readonly value: TaskOutcomeValue
  readonly verifierClaim: VerifierClaim
  readonly evidence: readonly EvidenceRef[]
  readonly note?: string
  readonly scores?: readonly TaskOutcomeScore[]
}

export interface IntentHashResult {
  readonly intent: NormalizedFinalizationIntent
  readonly intentHash: ContentAddress
}

function isControlChar(ch: string): boolean {
  const cp = ch.codePointAt(0)!
  return (cp >= 0x00 && cp <= 0x1f) || (cp >= 0x7f && cp <= 0x9f)
}

function assertNoControl(label: string, value: string): void {
  for (const ch of value) {
    if (isControlChar(ch)) {
      throw new TaskOutcomeFinalizationValidationError(
        `${label} must not contain Unicode control characters`,
      )
    }
  }
}

function assertTrimmedId(label: string, raw: unknown, maxLen: number): string {
  if (typeof raw !== 'string') {
    throw new TaskOutcomeFinalizationValidationError(`${label} must be a string`)
  }
  const trimmed = raw.trim()
  if (trimmed.length < 1 || trimmed.length > maxLen) {
    throw new TaskOutcomeFinalizationValidationError(
      `${label} must be 1–${maxLen} characters after trim`,
    )
  }
  assertNoControl(label, trimmed)
  return trimmed
}

function assertSha256Hash(label: string, hash: unknown): ContentAddress {
  if (typeof hash !== 'string' || !SHA256_RE.test(hash)) {
    throw new TaskOutcomeFinalizationValidationError(
      `${label} must be sha256:<64 lowercase hex>`,
    )
  }
  return hash as ContentAddress
}

function evidenceSortKey(ref: EvidenceRef): string {
  if (ref.kind === 'event') return `event\0${ref.eventId}`
  return `object\0${ref.objectId}\0${ref.hash}`
}

function assertJsonSafe(label: string, value: unknown): void {
  try {
    canonicalize(value)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    throw new TaskOutcomeFinalizationValidationError(`${label} is not JSON-safe: ${msg}`)
  }
}

function normalizeEvidence(raw: unknown): readonly EvidenceRef[] {
  if (!Array.isArray(raw)) {
    throw new TaskOutcomeFinalizationValidationError('evidence must be an array')
  }
  if (raw.length < 1) {
    throw new TaskOutcomeFinalizationValidationError('evidence must contain at least 1 ref')
  }
  if (raw.length > 128) {
    throw new TaskOutcomeFinalizationValidationError('evidence must contain at most 128 refs')
  }

  const normalized: EvidenceRef[] = []
  for (let i = 0; i < raw.length; i++) {
    const item = raw[i]
    if (item === null || typeof item !== 'object' || Array.isArray(item)) {
      throw new TaskOutcomeFinalizationValidationError(`evidence[${i}] must be an object`)
    }
    const kind = (item as { kind?: unknown }).kind
    if (kind === 'event') {
      const eventId = assertTrimmedId(`evidence[${i}].eventId`, (item as { eventId?: unknown }).eventId, 256)
      normalized.push({ kind: 'event', eventId })
    } else if (kind === 'object') {
      const objectId = assertTrimmedId(`evidence[${i}].objectId`, (item as { objectId?: unknown }).objectId, 256)
      const hash = assertSha256Hash(`evidence[${i}].hash`, (item as { hash?: unknown }).hash)
      normalized.push({ kind: 'object', objectId, hash })
    } else {
      throw new TaskOutcomeFinalizationValidationError(
        `evidence[${i}].kind must be "event" or "object"`,
      )
    }
  }

  const sorted = [...normalized].sort((a, b) => {
    const ka = evidenceSortKey(a)
    const kb = evidenceSortKey(b)
    return ka < kb ? -1 : ka > kb ? 1 : 0
  })

  for (let i = 1; i < sorted.length; i++) {
    if (evidenceSortKey(sorted[i - 1]!) === evidenceSortKey(sorted[i]!)) {
      throw new TaskOutcomeFinalizationValidationError('evidence contains duplicate refs after normalization')
    }
  }

  assertJsonSafe('evidence', sorted)
  return sorted
}

function normalizeScores(raw: unknown): readonly TaskOutcomeScore[] | undefined {
  if (raw === undefined) return undefined
  if (!Array.isArray(raw)) {
    throw new TaskOutcomeFinalizationValidationError('scores must be an array when provided')
  }
  if (raw.length > 64) {
    throw new TaskOutcomeFinalizationValidationError('scores must contain at most 64 entries')
  }

  const out: TaskOutcomeScore[] = []
  const seen = new Set<string>()
  for (let i = 0; i < raw.length; i++) {
    const item = raw[i]
    if (item === null || typeof item !== 'object' || Array.isArray(item)) {
      throw new TaskOutcomeFinalizationValidationError(`scores[${i}] must be an object`)
    }
    const name = assertTrimmedId(`scores[${i}].name`, (item as { name?: unknown }).name, 128)
    if (seen.has(name)) {
      throw new TaskOutcomeFinalizationValidationError(`scores contain duplicate name "${name}"`)
    }
    seen.add(name)
    const value = (item as { value?: unknown }).value
    if (typeof value !== 'number' && typeof value !== 'string' && typeof value !== 'boolean') {
      throw new TaskOutcomeFinalizationValidationError(
        `scores[${i}].value must be number | string | boolean`,
      )
    }
    if (typeof value === 'number' && !Number.isFinite(value)) {
      throw new TaskOutcomeFinalizationValidationError(`scores[${i}].value must be a finite number`)
    }
    out.push({ name, value })
  }

  out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
  assertJsonSafe('scores', out)
  return out
}

function normalizeNote(raw: unknown): string | undefined {
  if (raw === undefined) return undefined
  if (typeof raw !== 'string') {
    throw new TaskOutcomeFinalizationValidationError('note must be a string when provided')
  }
  // note is NOT trimmed; count Unicode code points
  let cps = 0
  for (const _ of raw) cps++
  if (cps > 8192) {
    throw new TaskOutcomeFinalizationValidationError('note must be at most 8192 Unicode code points')
  }
  assertJsonSafe('note', raw)
  return raw
}

function normalizeVerifierClaim(raw: unknown): VerifierClaim {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new TaskOutcomeFinalizationValidationError('verifierClaim must be an object')
  }
  const type = (raw as { type?: unknown }).type
  if (type !== 'human' && type !== 'eval' && type !== 'rule' && type !== 'service') {
    throw new TaskOutcomeFinalizationValidationError(
      'verifierClaim.type must be human | eval | rule | service',
    )
  }
  const id = assertTrimmedId('verifierClaim.id', (raw as { id?: unknown }).id, 256)
  const claim: VerifierClaim = { type, id }
  assertJsonSafe('verifierClaim', claim)
  return claim
}

function normalizeValue(raw: unknown): TaskOutcomeValue {
  if (raw !== 'success' && raw !== 'failure' && raw !== 'partial' && raw !== 'unknown') {
    throw new TaskOutcomeFinalizationValidationError(
      'value must be success | failure | partial | unknown',
    )
  }
  return raw
}

/**
 * Normalize finalize input fields that participate in intentHash.
 * finalizationId / expectedState / finalizedAt are intentionally excluded.
 */
export function normalizeFinalizationIntent(input: {
  runId: unknown
  value: unknown
  verifierClaim: unknown
  evidence: unknown
  note?: unknown
  scores?: unknown
}): IntentHashResult {
  const runId = assertTrimmedId('runId', input.runId, 256)
  const value = normalizeValue(input.value)
  const verifierClaim = normalizeVerifierClaim(input.verifierClaim)
  const evidence = normalizeEvidence(input.evidence)
  const note = normalizeNote(input.note)
  const scores = normalizeScores(input.scores)

  const intent: NormalizedFinalizationIntent = {
    schemaVersion: 1,
    runId,
    value,
    verifierClaim,
    evidence,
    ...(note !== undefined ? { note } : {}),
    ...(scores !== undefined ? { scores } : {}),
  }

  const intentHash = hashCanonical(intent) as ContentAddress
  return { intent, intentHash }
}

export function assertFinalizationId(raw: unknown): string {
  return assertTrimmedId('finalizationId', raw, 128)
}

export function buildRecordWithoutHash(args: {
  intent: NormalizedFinalizationIntent
  finalizationId: string
  intentHash: ContentAddress
  finalizedAt: number
}): Omit<TaskOutcomeFinalization, 'recordHash'> {
  if (!Number.isSafeInteger(args.finalizedAt) || args.finalizedAt < 0) {
    throw new TaskOutcomeFinalizationValidationError('finalizedAt must be a non-negative safe integer')
  }
  return {
    ...args.intent,
    state: 'finalized',
    finalizationId: args.finalizationId,
    intentHash: args.intentHash,
    finalizedAt: args.finalizedAt,
  }
}

export function computeRecordHash(
  recordWithoutHash: Omit<TaskOutcomeFinalization, 'recordHash'>,
): ContentAddress {
  return hashCanonical(recordWithoutHash) as ContentAddress
}

export function assembleFinalization(
  recordWithoutHash: Omit<TaskOutcomeFinalization, 'recordHash'>,
): TaskOutcomeFinalization {
  const recordHash = computeRecordHash(recordWithoutHash)
  return { ...recordWithoutHash, recordHash }
}

export function canonicalizeFinalization(record: TaskOutcomeFinalization): string {
  return canonicalize(record)
}

const FINALIZATION_TOP_LEVEL_KEYS: Record<string, true> = {
  schemaVersion: true,
  state: true,
  runId: true,
  value: true,
  verifierClaim: true,
  evidence: true,
  note: true,
  scores: true,
  finalizationId: true,
  intentHash: true,
  finalizedAt: true,
  recordHash: true,
}

/**
 * Validate a stored finalization snapshot: strict top-level schema, runId match,
 * recordHash. Unknown top-level fields fail closed (corruption path via store).
 * Returns a deep-cloned independent snapshot.
 */
export function parseAndValidateFinalization(
  raw: unknown,
  expectedRunId?: string,
): TaskOutcomeFinalization {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new TaskOutcomeFinalizationValidationError('finalization record must be an object')
  }
  const r = raw as Record<string, unknown>

  const unknownKeys = Object.keys(r)
    .filter((k) => FINALIZATION_TOP_LEVEL_KEYS[k] !== true)
    .sort()
  if (unknownKeys.length > 0) {
    throw new TaskOutcomeFinalizationValidationError(
      `finalization record contains unknown top-level field(s): ${unknownKeys.join(', ')}`,
    )
  }

  if (r.schemaVersion !== 1) {
    throw new TaskOutcomeFinalizationValidationError(
      `unsupported finalization schemaVersion: ${String(r.schemaVersion)}`,
    )
  }
  if (r.state !== 'finalized') {
    throw new TaskOutcomeFinalizationValidationError(
      `finalization state must be "finalized", got ${String(r.state)}`,
    )
  }

  const { intent, intentHash } = normalizeFinalizationIntent({
    runId: r.runId,
    value: r.value,
    verifierClaim: r.verifierClaim,
    evidence: r.evidence,
    note: r.note,
    scores: r.scores,
  })

  if (expectedRunId !== undefined && intent.runId !== expectedRunId) {
    throw new TaskOutcomeFinalizationValidationError(
      `finalization runId mismatch: stored "${intent.runId}" vs requested "${expectedRunId}"`,
    )
  }

  const finalizationId = assertFinalizationId(r.finalizationId)
  const storedIntentHash = assertSha256Hash('intentHash', r.intentHash)
  if (storedIntentHash !== intentHash) {
    throw new TaskOutcomeFinalizationValidationError(
      'finalization intentHash does not match normalized intent',
    )
  }

  if (typeof r.finalizedAt !== 'number' || !Number.isSafeInteger(r.finalizedAt) || r.finalizedAt < 0) {
    throw new TaskOutcomeFinalizationValidationError('finalizedAt must be a non-negative safe integer')
  }

  const storedRecordHash = assertSha256Hash('recordHash', r.recordHash)
  const withoutHash = buildRecordWithoutHash({
    intent,
    finalizationId,
    intentHash,
    finalizedAt: r.finalizedAt,
  })
  const expectedRecordHash = computeRecordHash(withoutHash)
  if (storedRecordHash !== expectedRecordHash) {
    throw new TaskOutcomeFinalizationValidationError(
      'finalization recordHash mismatch (record may be corrupted)',
    )
  }

  // Independent deep snapshot via canonicalize round-trip
  const assembled: TaskOutcomeFinalization = { ...withoutHash, recordHash: expectedRecordHash }
  return JSON.parse(canonicalize(assembled)) as TaskOutcomeFinalization
}

export function snapshotFinalization(record: TaskOutcomeFinalization): TaskOutcomeFinalization {
  return JSON.parse(canonicalizeFinalization(record)) as TaskOutcomeFinalization
}
