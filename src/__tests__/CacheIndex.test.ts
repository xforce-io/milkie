import { CacheIndex, CacheIndexEmptyError } from '../trace/CacheIndex'
import type { Event, ToolRespondedPayload } from '../trace/types'
import { LLM_OUTCOME_SCHEMA_VERSION } from '../trace/types'
import type { ModelResponse } from '../types/model'
import { ModelGatewayError } from '../gateway/ModelGatewayError'
import { IOControlError, LlmInvocationError } from '../types/model'
import { TraceIntegrityError } from '../trace/TraceIntegrityError'
import { SAFE_MESSAGES } from '../gateway/ModelGatewayError'

const llmResp = (text: string): ModelResponse => ({
  content:      [{ type: 'text', text }],
  toolCalls:    [],
  finishReason: 'end_turn',
})

/** Legacy success terminal (no status) — still accepted by §8.6. */
const mkLegacyLlmResponded = (hash: string, text: string, id?: string): Event => ({
  id:        id ?? `e-${Math.random()}`,
  runId:     'r1',
  type:      'llm.responded',
  actor:     'runtime',
  timestamp: 1,
  payload:   { response: llmResp(text), requestHash: hash },
})

const mkV2Ok = (hash: string, text: string, causedBy: string, id?: string): Event => ({
  id:        id ?? `ok-${Math.random()}`,
  runId:     'r1',
  type:      'llm.responded',
  actor:     'runtime',
  timestamp: 1,
  causedBy,
  payload:   { status: 'ok' as const, response: llmResp(text), requestHash: hash },
})

const mkV2Err = (
  hash: string,
  causedBy: string,
  error: Record<string, unknown>,
  id?: string,
): Event => ({
  id:        id ?? `err-${Math.random()}`,
  runId:     'r1',
  type:      'llm.responded',
  actor:     'runtime',
  timestamp: 1,
  causedBy,
  payload:   { status: 'error' as const, error, requestHash: hash },
})

const mkV2Req = (hash: string, id: string, indexHint = 0): Event => ({
  id,
  runId: 'r1',
  type: 'llm.requested',
  actor: 'runtime',
  timestamp: indexHint,
  payload: {
    request: { model: 'm', messages: [] },
    requestHash: hash,
    outcomeSchemaVersion: LLM_OUTCOME_SCHEMA_VERSION,
  },
})

const mkLegacyReq = (hash: string, id: string): Event => ({
  id,
  runId: 'r1',
  type: 'llm.requested',
  actor: 'runtime',
  timestamp: 1,
  payload: {
    request: { model: 'm', messages: [] },
    requestHash: hash,
  },
})

const mkToolResponded = (hash: string, output?: unknown, error?: NonNullable<ToolRespondedPayload['error']>): Event<ToolRespondedPayload> => ({
  id:        `e-${Math.random()}`,
  runId:     'r1',
  type:      'tool.responded',
  actor:     'runtime',
  timestamp: 1,
  payload:   { toolName: 't', output, error, requestHash: hash },
})

const modelEnv = {
  code: 'MODEL_RATE_LIMITED' as const,
  message: SAFE_MESSAGES.MODEL_RATE_LIMITED,
  phase: 'request' as const,
  provider: 'anthropic',
  model: 'm',
  retryable: true,
}

describe('CacheIndex', () => {
  it('fromEvents builds empty index for empty events', () => {
    const idx = CacheIndex.fromEvents([])
    expect(idx.remaining()).toEqual({ llm: 0, tool: 0, clock: 0, uuid: 0 })
  })

  it('consumeLLM serves legacy cached responses in FIFO order per hash', () => {
    const idx = CacheIndex.fromEvents([
      mkLegacyLlmResponded('h1', 'a', 'a1'),
      mkLegacyLlmResponded('h1', 'b', 'a2'),
      mkLegacyLlmResponded('h2', 'c', 'a3'),
    ])
    expect(idx.consumeLLM('h1').content[0]).toMatchObject({ text: 'a' })
    expect(idx.consumeLLM('h1').content[0]).toMatchObject({ text: 'b' })
    expect(idx.consumeLLM('h2').content[0]).toMatchObject({ text: 'c' })
  })

  it('consumeLLM throws when queue exhausted', () => {
    const idx = CacheIndex.fromEvents([mkLegacyLlmResponded('h', 'x', 'x1')])
    idx.consumeLLM('h')
    expect(() => idx.consumeLLM('h')).toThrow(CacheIndexEmptyError)
  })

  it('consumeLLM throws when hash never seen', () => {
    const idx = CacheIndex.fromEvents([])
    expect(() => idx.consumeLLM('missing')).toThrow(CacheIndexEmptyError)
  })

  it('consumeTool returns output for successful tool', () => {
    const idx = CacheIndex.fromEvents([mkToolResponded('th', { ok: 1 })])
    expect(idx.consumeTool('th')).toEqual({ ok: 1 })
  })

  it('consumeTool rethrows Error with retryable/code/name preserved', () => {
    const idx = CacheIndex.fromEvents([
      mkToolResponded('th', undefined, { message: 'boom', retryable: true, code: 'E', name: 'ToolBoom' }),
    ])
    try {
      idx.consumeTool('th')
      throw new Error('expected throw')
    } catch (err) {
      const e = err as Error & { retryable?: boolean; code?: string }
      expect(e).toBeInstanceOf(Error)
      expect(e.message).toBe('boom')
      expect(e.retryable).toBe(true)
      expect(e.code).toBe('E')
      expect(e.name).toBe('ToolBoom')
    }
  })

  it('remaining tracks unconsumed counts', () => {
    const idx = CacheIndex.fromEvents([
      mkLegacyLlmResponded('h', 'a', 'l1'),
      mkToolResponded('t', 1),
      { id: 'c1', runId: 'r1', type: 'clock.read', actor: 'a', timestamp: 1, payload: { value: 9 } },
      { id: 'u1', runId: 'r1', type: 'uuid.generated', actor: 'a', timestamp: 1, payload: { value: 'u' } },
    ])
    expect(idx.remaining()).toEqual({ llm: 1, tool: 1, clock: 1, uuid: 1 })
    idx.consumeLLM('h')
    expect(idx.remaining().llm).toBe(0)
  })

  it('allHashes returns all unique hashes seen', () => {
    const idx = CacheIndex.fromEvents([
      mkLegacyLlmResponded('h1', 'a', 'l1'),
      mkLegacyLlmResponded('h2', 'b', 'l2'),
      mkToolResponded('t1', 1),
    ])
    expect(idx.allHashes().llm.sort()).toEqual(['h1', 'h2'])
    expect(idx.allHashes().tool).toEqual(['t1'])
  })

  it('v2 request order wins over terminal completion order for same hash', () => {
    const events: Event[] = [
      mkV2Req('h', 'req-1', 1),
      mkV2Req('h', 'req-2', 2),
      // terminal for req-2 arrives first
      mkV2Ok('h', 'second', 'req-2', 'term-2'),
      mkV2Ok('h', 'first', 'req-1', 'term-1'),
    ]
    const idx = CacheIndex.fromEvents(events)
    expect(idx.consumeLLM('h').content[0]).toMatchObject({ text: 'first' })
    expect(idx.consumeLLM('h').content[0]).toMatchObject({ text: 'second' })
  })

  it('consumeLLM reconstructs ModelGatewayError from v2 failure terminal', () => {
    const events: Event[] = [
      mkV2Req('h', 'req-1'),
      mkV2Err('h', 'req-1', modelEnv, 'term-1'),
    ]
    const idx = CacheIndex.fromEvents(events)
    expect(() => idx.consumeLLM('h')).toThrow(ModelGatewayError)
    try {
      CacheIndex.fromEvents(events).consumeLLM('h')
    } catch (err) {
      expect(err).toBeInstanceOf(ModelGatewayError)
      expect((err as ModelGatewayError).envelope).toEqual(modelEnv)
      expect((err as ModelGatewayError).name).toBe('ModelGatewayError')
    }
  })

  it('consumeLLM reconstructs IOControlError and LlmInvocationError', () => {
    const controlEnv = {
      code: 'IO_CANCELLED' as const,
      message: 'I/O invocation was cancelled.' as const,
      phase: 'io_control' as const,
      operation: 'llm' as const,
      retryable: false as const,
    }
    const genericEnv = {
      code: 'LLM_INVOCATION_FAILED' as const,
      message: 'LLM invocation failed.' as const,
      phase: 'ioport' as const,
      model: 'm',
      retryable: false as const,
    }
    const controlIdx = CacheIndex.fromEvents([
      mkV2Req('hc', 'rc'),
      mkV2Err('hc', 'rc', controlEnv, 'tc'),
    ])
    expect(() => controlIdx.consumeLLM('hc')).toThrow(IOControlError)

    const genericIdx = CacheIndex.fromEvents([
      mkV2Req('hg', 'rg'),
      mkV2Err('hg', 'rg', genericEnv, 'tg'),
    ])
    expect(() => genericIdx.consumeLLM('hg')).toThrow(LlmInvocationError)
  })

  it('duplicate_event_id is reported with the colliding id', () => {
    expect(() => CacheIndex.fromEvents([
      mkLegacyLlmResponded('h', 'a', 'same'),
      mkLegacyLlmResponded('h', 'b', 'same'),
    ])).toThrow(expect.objectContaining({ name: 'TraceIntegrityError', kind: 'duplicate_event_id', eventId: 'same' }))
  })

  it('dangling v2 request is integrity error', () => {
    expect(() => CacheIndex.fromEvents([mkV2Req('h', 'req-only')])).toThrow(
      expect.objectContaining({ kind: 'dangling_request', requestEventId: 'req-only' }),
    )
  })

  it('orphan terminal with unknown causedBy is integrity error', () => {
    expect(() => CacheIndex.fromEvents([
      mkV2Ok('h', 'x', 'missing-req', 'term'),
    ])).toThrow(expect.objectContaining({ kind: 'orphan_terminal', eventId: 'term' }))
  })

  it('hash_mismatch between request and terminal is integrity error', () => {
    expect(() => CacheIndex.fromEvents([
      mkV2Req('h1', 'req-1'),
      mkV2Ok('h2', 'x', 'req-1', 'term-1'),
    ])).toThrow(expect.objectContaining({ kind: 'hash_mismatch' }))
  })

  it('duplicate_terminal for two terminals claiming one request', () => {
    expect(() => CacheIndex.fromEvents([
      mkV2Req('h', 'req-1'),
      mkV2Ok('h', 'a', 'req-1', 'term-1'),
      mkV2Ok('h', 'b', 'req-1', 'term-2'),
    ])).toThrow(expect.objectContaining({ kind: 'duplicate_terminal' }))
  })

  it('malformed failure envelope is malformed_payload', () => {
    expect(() => CacheIndex.fromEvents([
      mkV2Req('h', 'req-1'),
      mkV2Err('h', 'req-1', {
        code: 'MODEL_TIMEOUT',
        message: 'SECRET raw provider text',
        phase: 'request',
        provider: 'anthropic',
        model: 'm',
        retryable: true,
      }, 'term-1'),
    ])).toThrow(expect.objectContaining({ kind: 'malformed_payload' }))
  })

  it('legacy terminal-only success (no request) still enqueues', () => {
    const idx = CacheIndex.fromEvents([mkLegacyLlmResponded('solo', 'ok', 'solo-term')])
    expect(idx.consumeLLM('solo').content[0]).toMatchObject({ text: 'ok' })
  })

  it('legacy request + two no-causedBy terminals → second is duplicate_terminal', () => {
    expect(() => CacheIndex.fromEvents([
      mkLegacyReq('h', 'lreq'),
      mkLegacyLlmResponded('h', 'a', 't1'),
      mkLegacyLlmResponded('h', 'b', 't2'),
    ])).toThrow(expect.objectContaining({ kind: 'duplicate_terminal', eventId: 't2' }))
  })

  it('ambiguous_legacy when no-causedBy terminal collides with v2 hash', () => {
    expect(() => CacheIndex.fromEvents([
      mkV2Req('h', 'v2req'),
      mkV2Ok('h', 'ok', 'v2req', 'v2term'),
      mkLegacyLlmResponded('h', 'extra', 'legacy-term'),
    ])).toThrow(expect.objectContaining({ kind: 'ambiguous_legacy', eventId: 'legacy-term' }))
  })

  it('Phase 2 no-hash events are skipped and do not poison the index', () => {
    const idx = CacheIndex.fromEvents([
      {
        id: 'p2-req', runId: 'r1', type: 'llm.requested', actor: 'a', timestamp: 1,
        payload: { request: { model: 'm', messages: [] } },
      } as Event,
      {
        id: 'p2-term', runId: 'r1', type: 'llm.responded', actor: 'a', timestamp: 2,
        payload: { response: llmResp('old') },
      } as Event,
      mkLegacyLlmResponded('h', 'new', 'ok1'),
    ])
    expect(idx.consumeLLM('h').content[0]).toMatchObject({ text: 'new' })
    expect(idx.allHashes().llm).toEqual(['h'])
  })

  it('v2 request with empty requestHash is malformed_payload not Phase2 skip', () => {
    expect(() => CacheIndex.fromEvents([
      {
        id: 'bad-v2',
        runId: 'r1',
        type: 'llm.requested',
        actor: 'a',
        timestamp: 1,
        payload: {
          request: { model: 'm', messages: [] },
          requestHash: '',
          outcomeSchemaVersion: LLM_OUTCOME_SCHEMA_VERSION,
        },
      },
    ])).toThrow(expect.objectContaining({ kind: 'malformed_payload', eventId: 'bad-v2' }))
  })

  it('v2 request missing requestHash is malformed_payload at index build', () => {
    expect(() => CacheIndex.fromEvents([
      {
        id: 'bad-v2-nohash',
        runId: 'r1',
        type: 'llm.requested',
        actor: 'a',
        timestamp: 1,
        payload: {
          request: { model: 'm', messages: [] },
          outcomeSchemaVersion: LLM_OUTCOME_SCHEMA_VERSION,
        },
      },
    ])).toThrow(expect.objectContaining({ kind: 'malformed_payload', eventId: 'bad-v2-nohash' }))
  })

  it('v2 request with unknown outcomeSchemaVersion is malformed_payload', () => {
    expect(() => CacheIndex.fromEvents([
      {
        id: 'bad-ver',
        runId: 'r1',
        type: 'llm.requested',
        actor: 'a',
        timestamp: 1,
        payload: {
          request: { model: 'm', messages: [] },
          requestHash: 'h',
          outcomeSchemaVersion: 99,
        },
      },
    ])).toThrow(expect.objectContaining({ kind: 'malformed_payload', eventId: 'bad-ver' }))
  })

  it('model failure provider outside closed set is malformed_payload', () => {
    expect(() => CacheIndex.fromEvents([
      mkV2Req('h', 'req-1'),
      mkV2Err('h', 'req-1', {
        ...modelEnv,
        provider: 'evil-provider',
      }, 'term-1'),
    ])).toThrow(expect.objectContaining({ kind: 'malformed_payload', eventId: 'term-1' }))
  })
})
