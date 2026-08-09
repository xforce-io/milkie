import { ModelGatewayError, SAFE_MESSAGES } from '../gateway/ModelGatewayError.js'
import {
  IOControlError,
  LlmInvocationError,
  type IOControlErrorCode,
  type IOControlErrorEnvelope,
  type LlmInvocationFailureEnvelope,
  type ModelErrorCode,
  type ModelErrorEnvelope,
  type ModelErrorPhase,
  type ModelRequest,
  type ModelResponse,
} from '../types/model.js'
import { TraceIntegrityError, type TraceIntegrityErrorKind } from './TraceIntegrityError.js'
import type {
  Event,
  LlmCacheStats,
  LlmRespondedPayload,
  RecordedLlmFailureEnvelope,
  TrustedProviderFamily,
} from './types.js'
import { LLM_OUTCOME_SCHEMA_VERSION } from './types.js'

const SAFE_ID_RE = /^[A-Za-z0-9._:/-]{1,128}$/
const MODEL_ERROR_CODES: Record<ModelErrorCode, true> = {
  MODEL_CONNECTION_ERROR: true,
  MODEL_TIMEOUT: true,
  MODEL_RATE_LIMITED: true,
  MODEL_AUTH_ERROR: true,
  MODEL_BAD_RESPONSE: true,
  MODEL_UNKNOWN_ERROR: true,
}
const MODEL_ERROR_PHASES: Record<ModelErrorPhase, true> = {
  request: true,
  stream_open: true,
  stream_read: true,
  response_parse: true,
}
const CONTROL_CODES: Record<IOControlErrorCode, true> = {
  IO_CANCELLED: true,
  IO_DEADLINE_EXCEEDED: true,
}
const TRUSTED_PROVIDERS: Record<TrustedProviderFamily, true> = {
  anthropic: true,
  'openai-compatible': true,
}
const CONTROL_MESSAGES = {
  IO_CANCELLED: 'I/O invocation was cancelled.',
  IO_DEADLINE_EXCEEDED: 'I/O invocation deadline exceeded.',
} as const satisfies Record<IOControlErrorCode, IOControlErrorEnvelope['message']>

const GENERIC_MESSAGE = 'LLM invocation failed.' as const

export type LlmOutcomeStatus = 'ok' | 'error'

export interface LlmSuccessOutcome {
  status: 'ok'
  response: ModelResponse
  requestHash: string
  cacheStats?: LlmCacheStats
  /** true when decoded from a legacy terminal (no status field). */
  legacy?: boolean
}

export interface LlmFailureOutcome {
  status: 'error'
  error: RecordedLlmFailureEnvelope
  requestHash: string
}

export type LlmOutcome = LlmSuccessOutcome | LlmFailureOutcome

/** Safe consumer view of a recorded LLM failure (already validated). */
export interface LlmFailureView {
  code: string
  message: string
  phase: string
  retryable: boolean
  provider?: string
  model?: string
  status?: number
}

export interface TrustedProviderContext {
  /** Closed provider family from GatewayFactory; custom/injected → omit/unknown. */
  providerFamily?: TrustedProviderFamily | 'unknown'
}

export function isSafeIdentifier(value: unknown): value is string {
  return typeof value === 'string' && SAFE_ID_RE.test(value)
}

export function safeModelFromRequest(request: ModelRequest | undefined): string {
  const model = request?.model
  return isSafeIdentifier(model) ? model : 'unknown'
}

export function trustedProviderOf(ctx?: TrustedProviderContext): TrustedProviderFamily | 'unknown' {
  const family = ctx?.providerFamily
  if (family && family !== 'unknown' && TRUSTED_PROVIDERS[family]) {
    return family
  }
  return 'unknown'
}

export function failureViewOf(error: RecordedLlmFailureEnvelope): LlmFailureView {
  return {
    code: error.code,
    message: error.message,
    phase: error.phase,
    retryable: error.retryable,
    ...('provider' in error && error.provider !== undefined ? { provider: error.provider } : {}),
    ...('model' in error && error.model !== undefined ? { model: error.model } : {}),
    ...('status' in error && typeof (error as ModelErrorEnvelope).status === 'number'
      ? { status: (error as ModelErrorEnvelope).status }
      : {}),
  }
}

function genericEnvelope(model: string): LlmInvocationFailureEnvelope {
  return {
    code: 'LLM_INVOCATION_FAILED',
    message: GENERIC_MESSAGE,
    phase: 'ioport',
    model: isSafeIdentifier(model) ? model : 'unknown',
    retryable: false,
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function isBoolean(value: unknown): value is boolean {
  return value === true || value === false
}

function isValidHttpStatus(value: unknown): value is number {
  return typeof value === 'number'
    && Number.isInteger(value)
    && value >= 100
    && value <= 599
}

/**
 * Accept only a real ModelGatewayError with closed code/phase and rebuild a
 * safe envelope. Free message/provider/model from the public error are ignored.
 */
export function sanitizeModelFailure(
  error: unknown,
  request: ModelRequest,
  trustedContext?: TrustedProviderContext,
): ModelErrorEnvelope | undefined {
  if (!(error instanceof ModelGatewayError)) return undefined
  const env = error.envelope
  if (!env || typeof env !== 'object') return undefined
  if (!MODEL_ERROR_CODES[env.code as ModelErrorCode]) return undefined
  if (!MODEL_ERROR_PHASES[env.phase as ModelErrorPhase]) return undefined
  if (!isBoolean(env.retryable)) return undefined
  if (env.status !== undefined && !isValidHttpStatus(env.status)) return undefined

  const code = env.code as ModelErrorCode
  const phase = env.phase as ModelErrorPhase
  return {
    code,
    message: SAFE_MESSAGES[code],
    phase,
    provider: trustedProviderOf(trustedContext),
    model: safeModelFromRequest(request),
    retryable: env.retryable,
    ...(env.status !== undefined ? { status: env.status } : {}),
  }
}

/**
 * Accept only a real IOControlError for operation=llm and rebuild fixed fields.
 * Public envelope free-text is never trusted.
 */
export function sanitizeControlFailure(error: unknown): IOControlErrorEnvelope | undefined {
  if (!(error instanceof IOControlError)) return undefined
  if (error.operation !== 'llm') return undefined
  if (!CONTROL_CODES[error.code]) return undefined
  if (error.phase !== 'io_control') return undefined
  if (error.retryable !== false) return undefined
  const env = error.envelope
  if (!env || typeof env !== 'object') return undefined
  if (env.operation !== 'llm') return undefined
  if (env.phase !== 'io_control') return undefined
  if (env.retryable !== false) return undefined
  if (!CONTROL_CODES[env.code]) return undefined
  if (env.provider !== undefined || env.model !== undefined) return undefined

  const code = error.code
  return {
    code,
    message: CONTROL_MESSAGES[code],
    phase: 'io_control',
    operation: 'llm',
    retryable: false,
  }
}

/**
 * Normalize a thrown value into a stable recorded failure envelope.
 * Order is fixed: Model → control → generic.
 */
export function normalizeLlmFailure(
  error: unknown,
  request: ModelRequest,
  trustedContext?: TrustedProviderContext,
): RecordedLlmFailureEnvelope {
  const model = sanitizeModelFailure(error, request, trustedContext)
  if (model) return model
  const control = sanitizeControlFailure(error)
  if (control) return control
  return genericEnvelope(safeModelFromRequest(request))
}

export function reconstructLlmError(error: RecordedLlmFailureEnvelope): Error {
  if (error.code === 'LLM_INVOCATION_FAILED') {
    return new LlmInvocationError(error.model)
  }
  if (error.phase === 'io_control') {
    const control = error as IOControlErrorEnvelope
    return new IOControlError(control.code, 'llm')
  }
  return new ModelGatewayError(error as ModelErrorEnvelope)
}

function decodeModelFailure(raw: Record<string, unknown>): ModelErrorEnvelope {
  if (!MODEL_ERROR_CODES[raw.code as ModelErrorCode]) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  if (!MODEL_ERROR_PHASES[raw.phase as ModelErrorPhase]) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  if (!isBoolean(raw.retryable)) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  // Provider is a closed set on the read path too (L2 §8.3): not free text.
  if (
    raw.provider !== 'anthropic'
    && raw.provider !== 'openai-compatible'
    && raw.provider !== 'unknown'
  ) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  if (typeof raw.model !== 'string' || !isSafeIdentifier(raw.model)) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  if (raw.status !== undefined && !isValidHttpStatus(raw.status)) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  const code = raw.code as ModelErrorCode
  if (raw.message !== SAFE_MESSAGES[code]) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  // Reject unexpected keys that would indicate tampering / extra secrets.
  for (const key of Object.keys(raw)) {
    if (!['code', 'message', 'phase', 'provider', 'model', 'retryable', 'status'].includes(key)) {
      throw new TraceIntegrityError({ kind: 'malformed_payload' })
    }
  }
  return {
    code,
    message: SAFE_MESSAGES[code],
    phase: raw.phase as ModelErrorPhase,
    provider: raw.provider,
    model: raw.model,
    retryable: raw.retryable,
    ...(raw.status !== undefined ? { status: raw.status as number } : {}),
  }
}

function decodeControlFailure(raw: Record<string, unknown>): IOControlErrorEnvelope {
  if (!CONTROL_CODES[raw.code as IOControlErrorCode]) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  if (raw.phase !== 'io_control') {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  if (raw.operation !== 'llm') {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  if (raw.retryable !== false) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  if (raw.provider !== undefined || raw.model !== undefined) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  const code = raw.code as IOControlErrorCode
  if (raw.message !== CONTROL_MESSAGES[code]) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  for (const key of Object.keys(raw)) {
    if (!['code', 'message', 'phase', 'operation', 'retryable', 'provider', 'model'].includes(key)) {
      throw new TraceIntegrityError({ kind: 'malformed_payload' })
    }
  }
  return {
    code,
    message: CONTROL_MESSAGES[code],
    phase: 'io_control',
    operation: 'llm',
    retryable: false,
  }
}

function decodeGenericFailure(raw: Record<string, unknown>): LlmInvocationFailureEnvelope {
  if (raw.code !== 'LLM_INVOCATION_FAILED') {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  if (raw.phase !== 'ioport') {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  if (raw.retryable !== false) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  if (raw.message !== GENERIC_MESSAGE) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  if (typeof raw.model !== 'string' || !isSafeIdentifier(raw.model)) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  if (raw.provider !== undefined) {
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  }
  for (const key of Object.keys(raw)) {
    if (!['code', 'message', 'phase', 'model', 'retryable', 'provider'].includes(key)) {
      throw new TraceIntegrityError({ kind: 'malformed_payload' })
    }
  }
  return {
    code: 'LLM_INVOCATION_FAILED',
    message: GENERIC_MESSAGE,
    phase: 'ioport',
    model: raw.model,
    retryable: false,
  }
}

/**
 * Strict decode of a persisted failure envelope. Any tamper → TraceIntegrityError.
 * Does NOT fall back to generic.
 */
export function decodeRecordedLlmFailure(
  raw: unknown,
  eventId?: string,
): RecordedLlmFailureEnvelope {
  if (!isPlainObject(raw)) {
    throw new TraceIntegrityError({ kind: 'malformed_payload', ...(eventId ? { eventId } : {}) })
  }
  try {
    if (raw.code === 'LLM_INVOCATION_FAILED') return decodeGenericFailure(raw)
    if (raw.phase === 'io_control') return decodeControlFailure(raw)
    if (typeof raw.code === 'string' && raw.code.startsWith('MODEL_')) return decodeModelFailure(raw)
    throw new TraceIntegrityError({ kind: 'malformed_payload' })
  } catch (err) {
    if (err instanceof TraceIntegrityError) {
      if (eventId && err.eventId === undefined) {
        throw new TraceIntegrityError({ kind: err.kind, eventId, requestEventId: err.requestEventId })
      }
      throw err
    }
    throw new TraceIntegrityError({ kind: 'malformed_payload', ...(eventId ? { eventId } : {}) })
  }
}

function isModelResponse(value: unknown): value is ModelResponse {
  if (!isPlainObject(value)) return false
  if (!Array.isArray(value.content)) return false
  if (!Array.isArray(value.toolCalls)) return false
  return true
}

function isCacheStats(value: unknown): value is LlmCacheStats {
  if (!isPlainObject(value)) return false
  return typeof value.readTokens === 'number'
    && typeof value.creationTokens === 'number'
    && typeof value.totalInputTokens === 'number'
    && typeof value.hitRate === 'number'
}

/**
 * Decode an llm.responded event payload into a validated LlmOutcome.
 * Accepts unknown wire shapes; throws TraceIntegrityError on malformation.
 */
export function decodeLlmOutcome(
  event: Event,
  opts?: { requireRequestHash?: boolean },
): LlmOutcome {
  if (event.type !== 'llm.responded') {
    throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
  }
  const payload = event.payload as unknown
  if (!isPlainObject(payload)) {
    throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
  }
  const requestHash = payload.requestHash
  if (typeof requestHash !== 'string' || requestHash.length === 0) {
    if (opts?.requireRequestHash === false && requestHash === undefined) {
      // Phase 2 no-hash terminals are not replay-indexable; callers skip them.
      throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
    }
    throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
  }

  const keys = Object.keys(payload)

  // v2 success
  if (payload.status === 'ok') {
    for (const key of keys) {
      if (!['status', 'response', 'requestHash', 'cacheStats'].includes(key)) {
        throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
      }
    }
    if (payload.error !== undefined) {
      throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
    }
    if (!isModelResponse(payload.response)) {
      throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
    }
    if (payload.cacheStats !== undefined && !isCacheStats(payload.cacheStats)) {
      throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
    }
    return {
      status: 'ok',
      response: payload.response,
      requestHash,
      ...(payload.cacheStats !== undefined ? { cacheStats: payload.cacheStats as LlmCacheStats } : {}),
    }
  }

  // v2 failure
  if (payload.status === 'error') {
    for (const key of keys) {
      if (!['status', 'error', 'requestHash'].includes(key)) {
        throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
      }
    }
    if (payload.response !== undefined || payload.cacheStats !== undefined) {
      throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
    }
    const error = decodeRecordedLlmFailure(payload.error, event.id)
    return { status: 'error', error, requestHash }
  }

  // status present but not ok/error
  if (payload.status !== undefined) {
    throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
  }

  // legacy success: no status, has response, no error
  for (const key of keys) {
    if (!['response', 'requestHash', 'cacheStats'].includes(key)) {
      throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
    }
  }
  if (payload.error !== undefined) {
    throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
  }
  if (!isModelResponse(payload.response)) {
    throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
  }
  if (payload.cacheStats !== undefined && !isCacheStats(payload.cacheStats)) {
    throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: event.id })
  }
  return {
    status: 'ok',
    response: payload.response,
    requestHash,
    legacy: true,
    ...(payload.cacheStats !== undefined ? { cacheStats: payload.cacheStats as LlmCacheStats } : {}),
  }
}

/** Build a v2 success terminal payload. */
export function buildSuccessTerminalPayload(
  response: ModelResponse,
  requestHash: string,
  cacheStats?: LlmCacheStats,
): LlmRespondedPayload {
  return {
    status: 'ok',
    response,
    requestHash,
    ...(cacheStats ? { cacheStats } : {}),
  }
}

/** Build a v2 failure terminal payload. */
export function buildFailureTerminalPayload(
  error: RecordedLlmFailureEnvelope,
  requestHash: string,
): LlmRespondedPayload {
  return {
    status: 'error',
    error,
    requestHash,
  }
}

export function isV2RequestPayload(payload: unknown): boolean {
  return isPlainObject(payload) && payload.outcomeSchemaVersion === LLM_OUTCOME_SCHEMA_VERSION
}

export { LLM_OUTCOME_SCHEMA_VERSION }

/** Decoder-backed safe view for LLM terminal rendering. Never carries raw wire secrets. */
export type LlmTerminalRenderView =
  | {
      status: 'ok'
      requestHash: string
      response: ModelResponse
      cacheStats?: LlmCacheStats
    }
  | {
      status: 'error'
      requestHash: string
      error: LlmFailureView
    }
  | {
      status: 'malformed'
      kind: TraceIntegrityErrorKind
      eventId: string
    }

/**
 * Project an llm.responded event into a render-safe payload.
 * Success/error go through the decoder; decode failure yields kind + eventId only.
 */
export function llmTerminalRenderView(event: Event): LlmTerminalRenderView {
  try {
    const outcome = decodeLlmOutcome(event)
    if (outcome.status === 'ok') {
      return {
        status: 'ok',
        requestHash: outcome.requestHash,
        response: outcome.response,
        ...(outcome.cacheStats !== undefined ? { cacheStats: outcome.cacheStats } : {}),
      }
    }
    return {
      status: 'error',
      requestHash: outcome.requestHash,
      error: failureViewOf(outcome.error),
    }
  } catch (err) {
    const kind = err instanceof TraceIntegrityError ? err.kind : 'malformed_payload'
    return { status: 'malformed', kind, eventId: event.id }
  }
}

/** Replace llm.responded.payload with a decoder-safe view for HTML/viewer embeds. */
export function sanitizeEventForRender(event: Event): Event {
  if (event.type !== 'llm.responded') return event
  return { ...event, payload: llmTerminalRenderView(event) }
}
