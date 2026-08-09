import {
  LlmInvocationError,
  TraceWriteError,
  TraceIntegrityError,
  IOControlError,
  decodeLlmOutcome,
  reconstructLlmError,
  LLM_OUTCOME_SCHEMA_VERSION,
} from '../index'
import type {
  LlmInvocationFailureEnvelope,
  TraceIntegrityErrorKind,
  RecordedLlmFailureEnvelope,
} from '../index'

describe('public barrel #229 exports', () => {
  it('exports typed errors with working instanceof', () => {
    const inv = new LlmInvocationError('m')
    expect(inv).toBeInstanceOf(LlmInvocationError)
    expect(inv.envelope.code).toBe('LLM_INVOCATION_FAILED')

    const write = new TraceWriteError({ stage: 'request', operation: 'llm', eventId: 'e1' })
    expect(write).toBeInstanceOf(TraceWriteError)
    expect(write.stage).toBe('request')

    const integ = new TraceIntegrityError({ kind: 'dangling_request', eventId: 'e2' })
    expect(integ).toBeInstanceOf(TraceIntegrityError)
    expect(integ.kind).toBe('dangling_request' satisfies TraceIntegrityErrorKind)

    // #228 identity preserved
    const ctrl = new IOControlError('IO_CANCELLED', 'llm')
    expect(ctrl).toBeInstanceOf(IOControlError)

    expect(LLM_OUTCOME_SCHEMA_VERSION).toBe(2)
    expect(typeof decodeLlmOutcome).toBe('function')
    expect(typeof reconstructLlmError).toBe('function')

    const env: LlmInvocationFailureEnvelope = inv.envelope
    const recorded: RecordedLlmFailureEnvelope = env
    expect(recorded.code).toBe('LLM_INVOCATION_FAILED')
  })
})
