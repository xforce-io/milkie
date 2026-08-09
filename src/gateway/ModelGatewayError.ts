import type {
  ModelErrorCode,
  ModelErrorEnvelope,
  ModelErrorPhase,
} from '../types/model.js'

export interface ModelErrorContext {
  provider: string
  model:    string
  phase:    ModelErrorPhase
}

export const SAFE_MESSAGES: Record<ModelErrorCode, string> = {
  MODEL_CONNECTION_ERROR: 'Model provider connection failed.',
  MODEL_TIMEOUT:          'Model provider request timed out.',
  MODEL_RATE_LIMITED:     'Model provider rate limit exceeded.',
  MODEL_AUTH_ERROR:       'Model provider authentication failed.',
  MODEL_BAD_RESPONSE:     'Model provider rejected the request or returned an invalid response.',
  MODEL_UNKNOWN_ERROR:    'Model provider request failed.',
}

function statusOf(error: unknown): number | undefined {
  if (!error || typeof error !== 'object') return undefined
  const status = (error as { status?: unknown }).status
  return typeof status === 'number' ? status : undefined
}

function namesOf(error: unknown): string {
  const names: string[] = []
  let current = error
  for (let depth = 0; depth < 4 && current && typeof current === 'object'; depth++) {
    const name = (current as { name?: unknown }).name
    if (typeof name === 'string') names.push(name)
    const ctorName = (current as { constructor?: { name?: unknown } }).constructor?.name
    if (typeof ctorName === 'string') names.push(ctorName)
    current = (current as { cause?: unknown }).cause
  }
  return names.join(' ').toLowerCase()
}

function transportCodeOf(error: unknown): string {
  let current = error
  for (let depth = 0; depth < 4 && current && typeof current === 'object'; depth++) {
    const code = (current as { code?: unknown }).code
    if (typeof code === 'string') return code.toUpperCase()
    current = (current as { cause?: unknown }).cause
  }
  return ''
}

function classify(error: unknown): { code: ModelErrorCode; retryable: boolean; status?: number } {
  const status = statusOf(error)
  const names = namesOf(error)
  const transportCode = transportCodeOf(error)

  if (
    names.includes('timeout') ||
    ['ETIMEDOUT', 'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_HEADERS_TIMEOUT', 'UND_ERR_BODY_TIMEOUT'].includes(transportCode)
  ) {
    return { code: 'MODEL_TIMEOUT', retryable: true, ...(status !== undefined ? { status } : {}) }
  }
  if (
    names.includes('connection') ||
    ['ECONNRESET', 'ECONNREFUSED', 'ENOTFOUND', 'EAI_AGAIN', 'EPIPE', 'UND_ERR_SOCKET'].includes(transportCode)
  ) {
    return { code: 'MODEL_CONNECTION_ERROR', retryable: true, ...(status !== undefined ? { status } : {}) }
  }
  if (status === 429) return { code: 'MODEL_RATE_LIMITED', retryable: true, status }
  if (status === 401 || status === 403) return { code: 'MODEL_AUTH_ERROR', retryable: false, status }
  if (status !== undefined) {
    return { code: 'MODEL_BAD_RESPONSE', retryable: status === 408 || status >= 500, status }
  }
  return { code: 'MODEL_UNKNOWN_ERROR', retryable: false }
}

export class ModelGatewayError extends Error {
  readonly envelope: ModelErrorEnvelope
  override readonly cause?: unknown

  constructor(envelope: ModelErrorEnvelope, cause?: unknown) {
    super(envelope.message)
    this.name = 'ModelGatewayError'
    this.envelope = { ...envelope }
    this.cause = cause
  }

  toJSON(): ModelErrorEnvelope {
    return { ...this.envelope }
  }
}

export function normalizeModelGatewayError(
  error: unknown,
  context: ModelErrorContext,
): ModelGatewayError {
  if (error instanceof ModelGatewayError) return error
  const initial = classify(error)
  const classified = initial.code === 'MODEL_UNKNOWN_ERROR' && context.phase === 'response_parse'
    ? { code: 'MODEL_BAD_RESPONSE' as const, retryable: false }
    : initial
  return new ModelGatewayError({
    code: classified.code,
    message: SAFE_MESSAGES[classified.code],
    phase: context.phase,
    provider: context.provider,
    model: context.model,
    retryable: classified.retryable,
    ...(classified.status !== undefined ? { status: classified.status } : {}),
  }, error)
}

export function modelErrorEnvelope(error: unknown): ModelErrorEnvelope | undefined {
  return error instanceof ModelGatewayError ? error.toJSON() : undefined
}
