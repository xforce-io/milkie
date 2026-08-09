import { ModelGatewayError, SAFE_MESSAGES } from '../gateway/ModelGatewayError'
import {
  IOControlError,
  LlmInvocationError,
  type ModelRequest,
} from '../types/model'
import {
  decodeLlmOutcome,
  decodeRecordedLlmFailure,
  llmTerminalRenderView,
  normalizeLlmFailure,
  reconstructLlmError,
  sanitizeControlFailure,
  sanitizeModelFailure,
} from '../trace/LlmOutcome'
import { TraceIntegrityError } from '../trace/TraceIntegrityError'
import type { Event } from '../trace/types'

const REQ: ModelRequest = { model: 'claude-test', messages: [] }

function modelErr(overrides: Partial<ConstructorParameters<typeof ModelGatewayError>[0]> = {}) {
  return new ModelGatewayError({
    code: 'MODEL_RATE_LIMITED',
    message: 'LEAKED secret provider body token=sk-live',
    phase: 'request',
    provider: 'evil-provider',
    model: 'evil-model',
    retryable: true,
    ...overrides,
  })
}

describe('LlmOutcome sanitizer', () => {
  it('rebuilds ModelGatewayError with safe message and trusted provider/model', () => {
    const env = sanitizeModelFailure(modelErr(), REQ, { providerFamily: 'anthropic' })
    expect(env).toEqual({
      code: 'MODEL_RATE_LIMITED',
      message: SAFE_MESSAGES.MODEL_RATE_LIMITED,
      phase: 'request',
      provider: 'anthropic',
      model: 'claude-test',
      retryable: true,
    })
    expect(JSON.stringify(env)).not.toContain('sk-live')
    expect(JSON.stringify(env)).not.toContain('evil')
  })

  it('falls back to unknown provider without trusted context', () => {
    const env = sanitizeModelFailure(modelErr(), REQ)
    expect(env?.provider).toBe('unknown')
  })

  it('rejects forged ModelGatewayError with illegal phase/code as undefined', () => {
    const forged = new ModelGatewayError({
      code: 'MODEL_TIMEOUT',
      message: SAFE_MESSAGES.MODEL_TIMEOUT,
      phase: 'io_control' as 'request',
      provider: 'x',
      model: 'y',
      retryable: true,
    })
    expect(sanitizeModelFailure(forged, REQ, { providerFamily: 'anthropic' })).toBeUndefined()
  })

  it('rebuilds control failure with fixed message and llm operation', () => {
    const err = new IOControlError('IO_DEADLINE_EXCEEDED', 'llm')
    // mutate public envelope-looking fields should not matter — sanitize rebuilds
    expect(sanitizeControlFailure(err)).toEqual({
      code: 'IO_DEADLINE_EXCEEDED',
      message: 'I/O invocation deadline exceeded.',
      phase: 'io_control',
      operation: 'llm',
      retryable: false,
    })
  })

  it('rejects control failure for tool operation', () => {
    expect(sanitizeControlFailure(new IOControlError('IO_CANCELLED', 'tool'))).toBeUndefined()
  })

  it('normalize order: model → control → generic; secrets never land', () => {
    const model = normalizeLlmFailure(modelErr({ code: 'MODEL_TIMEOUT', retryable: true }), REQ, {
      providerFamily: 'openai-compatible',
    })
    expect(model.code).toBe('MODEL_TIMEOUT')
    expect(model).toMatchObject({ provider: 'openai-compatible', message: SAFE_MESSAGES.MODEL_TIMEOUT })

    const control = normalizeLlmFailure(new IOControlError('IO_CANCELLED', 'llm'), REQ)
    expect(control).toMatchObject({ code: 'IO_CANCELLED', operation: 'llm' })

    const generic = normalizeLlmFailure(new Error('raw secret stack token=abc'), REQ)
    expect(generic).toEqual({
      code: 'LLM_INVOCATION_FAILED',
      message: 'LLM invocation failed.',
      phase: 'ioport',
      model: 'claude-test',
      retryable: false,
    })
    expect(JSON.stringify(generic)).not.toContain('secret')
    expect(JSON.stringify(generic)).not.toContain('abc')

    const fromString = normalizeLlmFailure('boom', { model: '!!!bad!!!', messages: [] })
    expect(fromString).toMatchObject({ code: 'LLM_INVOCATION_FAILED', model: 'unknown' })
  })
})

describe('LlmOutcome reconstruct', () => {
  it('reconstructs three typed error classes with stable envelopes', () => {
    const m = reconstructLlmError({
      code: 'MODEL_AUTH_ERROR',
      message: SAFE_MESSAGES.MODEL_AUTH_ERROR,
      phase: 'request',
      provider: 'anthropic',
      model: 'm',
      retryable: false,
    })
    expect(m).toBeInstanceOf(ModelGatewayError)
    expect(m.name).toBe('ModelGatewayError')
    expect((m as ModelGatewayError).envelope.code).toBe('MODEL_AUTH_ERROR')

    const c = reconstructLlmError({
      code: 'IO_CANCELLED',
      message: 'I/O invocation was cancelled.',
      phase: 'io_control',
      operation: 'llm',
      retryable: false,
    })
    expect(c).toBeInstanceOf(IOControlError)
    expect((c as IOControlError).envelope.code).toBe('IO_CANCELLED')

    const g = reconstructLlmError({
      code: 'LLM_INVOCATION_FAILED',
      message: 'LLM invocation failed.',
      phase: 'ioport',
      model: 'm',
      retryable: false,
    })
    expect(g).toBeInstanceOf(LlmInvocationError)
    expect((g as LlmInvocationError).envelope.code).toBe('LLM_INVOCATION_FAILED')
  })
})

describe('LlmOutcome decode', () => {
  const base = (payload: unknown, id = 't1'): Event => ({
    id, runId: 'r', type: 'llm.responded', actor: 'a', timestamp: 1, payload,
  })

  it('decodes v2 success and legacy success', () => {
    const v2 = decodeLlmOutcome(base({
      status: 'ok',
      response: { content: [{ type: 'text', text: 'hi' }], toolCalls: [] },
      requestHash: 'h',
    }))
    expect(v2).toMatchObject({ status: 'ok', requestHash: 'h' })

    const legacy = decodeLlmOutcome(base({
      response: { content: [], toolCalls: [] },
      requestHash: 'h2',
    }))
    expect(legacy).toMatchObject({ status: 'ok', legacy: true, requestHash: 'h2' })
  })

  it('decodes v2 failure and rejects tampered message', () => {
    const ok = decodeLlmOutcome(base({
      status: 'error',
      requestHash: 'h',
      error: {
        code: 'MODEL_TIMEOUT',
        message: SAFE_MESSAGES.MODEL_TIMEOUT,
        phase: 'request',
        provider: 'anthropic',
        model: 'm',
        retryable: true,
      },
    }))
    expect(ok.status).toBe('error')

    expect(() => decodeRecordedLlmFailure({
      code: 'MODEL_TIMEOUT',
      message: 'tampered',
      phase: 'request',
      provider: 'anthropic',
      model: 'm',
      retryable: true,
    })).toThrow(TraceIntegrityError)

    expect(() => decodeLlmOutcome(base({
      status: 'error',
      requestHash: 'h',
      error: {
        code: 'IO_CANCELLED',
        message: 'I/O invocation was cancelled.',
        phase: 'io_control',
        operation: 'llm',
        retryable: false,
        secret: 'nope',
      },
    }, 'bad'))).toThrow(expect.objectContaining({ kind: 'malformed_payload', eventId: 'bad' }))
  })

  it('rejects illegal status/response combinations', () => {
    expect(() => decodeLlmOutcome(base({
      status: 'ok',
      requestHash: 'h',
      response: { content: [], toolCalls: [] },
      error: { code: 'x' },
    }))).toThrow(TraceIntegrityError)

    expect(() => decodeLlmOutcome(base({
      status: 'error',
      requestHash: 'h',
      error: {
        code: 'LLM_INVOCATION_FAILED',
        message: 'LLM invocation failed.',
        phase: 'ioport',
        model: 'm',
        retryable: false,
      },
      response: { content: [], toolCalls: [] },
    }))).toThrow(TraceIntegrityError)
  })

  it('rejects model failure provider outside anthropic|openai-compatible|unknown', () => {
    expect(() => decodeRecordedLlmFailure({
      code: 'MODEL_TIMEOUT',
      message: SAFE_MESSAGES.MODEL_TIMEOUT,
      phase: 'request',
      provider: 'evil-provider',
      model: 'm',
      retryable: true,
    })).toThrow(expect.objectContaining({ kind: 'malformed_payload' }))

    expect(() => decodeRecordedLlmFailure({
      code: 'MODEL_TIMEOUT',
      message: SAFE_MESSAGES.MODEL_TIMEOUT,
      phase: 'request',
      provider: 'sk-live-looks-like-id',
      model: 'm',
      retryable: true,
    })).toThrow(expect.objectContaining({ kind: 'malformed_payload' }))

    // Closed set still accepts unknown.
    expect(decodeRecordedLlmFailure({
      code: 'MODEL_TIMEOUT',
      message: SAFE_MESSAGES.MODEL_TIMEOUT,
      phase: 'request',
      provider: 'unknown',
      model: 'm',
      retryable: true,
    })).toMatchObject({ provider: 'unknown' })
  })

  it('rejects unknown top-level keys on v2 ok/error and legacy terminals', () => {
    expect(() => decodeLlmOutcome(base({
      status: 'ok',
      requestHash: 'h',
      response: { content: [], toolCalls: [] },
      token: 'sk-secret-ok',
    }, 'ok-extra'))).toThrow(expect.objectContaining({ kind: 'malformed_payload', eventId: 'ok-extra' }))

    expect(() => decodeLlmOutcome(base({
      status: 'error',
      requestHash: 'h',
      error: {
        code: 'LLM_INVOCATION_FAILED',
        message: 'LLM invocation failed.',
        phase: 'ioport',
        model: 'm',
        retryable: false,
      },
      stack: 'SECRET stack',
      cause: { message: 'nope' },
    }, 'err-extra'))).toThrow(expect.objectContaining({ kind: 'malformed_payload', eventId: 'err-extra' }))

    expect(() => decodeLlmOutcome(base({
      response: { content: [], toolCalls: [] },
      requestHash: 'h-legacy',
      token: 'sk-legacy-secret',
    }, 'legacy-extra'))).toThrow(expect.objectContaining({ kind: 'malformed_payload', eventId: 'legacy-extra' }))
  })
})

describe('LlmOutcome render view', () => {
  const base = (payload: unknown, id = 't1'): Event => ({
    id, runId: 'r', type: 'llm.responded', actor: 'a', timestamp: 1, payload,
  })

  it('projects tampered terminal to kind/eventId only without secrets', () => {
    const SECRET = 'sk-render-view-secret'
    const view = llmTerminalRenderView(base({
      status: 'error',
      requestHash: 'h',
      error: {
        code: 'MODEL_TIMEOUT',
        message: SAFE_MESSAGES.MODEL_TIMEOUT,
        phase: 'request',
        provider: 'anthropic',
        model: 'm',
        retryable: true,
        stack: SECRET,
      },
      token: SECRET,
    }, 'bad-term'))
    expect(view).toEqual({
      status: 'malformed',
      kind: 'malformed_payload',
      eventId: 'bad-term',
    })
    expect(JSON.stringify(view)).not.toContain(SECRET)
  })

  it('projects validated error terminal to failure view fields only', () => {
    const view = llmTerminalRenderView(base({
      status: 'error',
      requestHash: 'h',
      error: {
        code: 'MODEL_TIMEOUT',
        message: SAFE_MESSAGES.MODEL_TIMEOUT,
        phase: 'request',
        provider: 'anthropic',
        model: 'm',
        retryable: true,
      },
    }))
    expect(view).toMatchObject({
      status: 'error',
      requestHash: 'h',
      error: {
        code: 'MODEL_TIMEOUT',
        message: SAFE_MESSAGES.MODEL_TIMEOUT,
        phase: 'request',
        provider: 'anthropic',
        model: 'm',
        retryable: true,
      },
    })
  })
})
