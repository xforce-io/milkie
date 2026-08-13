import { DefaultIOPort } from '../runtime/IOPort'
import { RecordingIOPort } from '../trace/RecordingIOPort'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { CausalCursor } from '../trace/CausalCursor'
import { TraceWriteError } from '../trace/TraceWriteError'
import { ModelGatewayError, SAFE_MESSAGES } from '../gateway/ModelGatewayError'
import {
  IOControlError,
  IOInvocationValidationError,
  LlmInvocationError,
  type IModelGateway,
  type ModelEvent,
  type ModelRequest,
  type ModelResponse,
} from '../types/model'
import type { IEventStore } from '../trace/EventStore'
import type { Event } from '../trace/types'
import { CacheIndex } from '../trace/CacheIndex'
import { ReplayingIOPort } from '../trace/ReplayingIOPort'
import { hashModelRequest } from '../trace/hash'

const REQ: ModelRequest = { model: 'm', messages: [] }
const OK: ModelResponse = {
  content: [{ type: 'text', text: 'ok' }],
  toolCalls: [],
  finishReason: 'end_turn',
}

class ScriptGateway implements IModelGateway {
  calls = 0
  constructor(private readonly impl: () => Promise<ModelResponse>) {}
  async complete(): Promise<ModelResponse> {
    this.calls++
    return this.impl()
  }
  async *stream(): AsyncIterable<ModelEvent> { yield* [] }
}

class CountingStore implements IEventStore {
  events: Event[] = []
  appendCalls = 0
  rejectOnCall?: number
  rejectAfterCommitOnCall?: number
  private call = 0

  async append(event: Event): Promise<void> {
    this.call++
    this.appendCalls = this.call
    if (this.rejectOnCall === this.call) {
      throw new Error('pre-commit reject')
    }
    this.events.push(event)
    if (this.rejectAfterCommitOnCall === this.call) {
      throw new Error('after-commit reject')
    }
  }
  async readByRunId(runId: string): Promise<Event[]> {
    return this.events.filter(e => e.runId === runId)
  }
  async readRange(runId: string, fromIndex: number, count?: number): Promise<Event[]> {
    const all = await this.readByRunId(runId)
    return all.slice(fromIndex, count === undefined ? undefined : fromIndex + count)
  }
}

describe('RecordingIOPort LLM failure terminals (#229)', () => {
  it('records ModelGatewayError as v2 error terminal and throws reconstructed error', async () => {
    const store = new MemoryEventStore()
    const cursor = new CausalCursor()
    const gateway = new ScriptGateway(async () => {
      throw new ModelGatewayError({
        code: 'MODEL_RATE_LIMITED',
        message: 'secret rate body token=xyz',
        phase: 'request',
        provider: 'leaked',
        model: 'leaked-model',
        retryable: true,
        status: 429,
      })
    })
    const port = new RecordingIOPort(
      new DefaultIOPort(gateway),
      store,
      'run-1',
      'runtime',
      undefined,
      cursor,
      'anthropic',
    )

    await expect(port.invokeLLM(REQ)).rejects.toBeInstanceOf(ModelGatewayError)
    const events = await store.readByRunId('run-1')
    expect(events.map(e => e.type)).toEqual(['llm.requested', 'llm.responded'])
    const req = events[0]!
    const term = events[1]!
    expect(req.payload).toMatchObject({ outcomeSchemaVersion: 2 })
    expect(term.causedBy).toBe(req.id)
    expect(term.payload).toEqual({
      status: 'error',
      requestHash: hashModelRequest(REQ),
      error: {
        code: 'MODEL_RATE_LIMITED',
        message: SAFE_MESSAGES.MODEL_RATE_LIMITED,
        phase: 'request',
        provider: 'anthropic',
        model: 'm',
        retryable: true,
        status: 429,
      },
    })
    expect(JSON.stringify(term.payload)).not.toContain('xyz')
    expect(cursor.lastLlmTerminalId).toBe(term.id)
    expect(cursor.lastLlmRespondedId).toBeUndefined()
    expect(cursor.lastIoEventId).toBe(term.id)
  })

  it('records control cancel/deadline and generic secret failures', async () => {
    for (const [label, thrower, code] of [
      ['cancel', async () => { throw new IOControlError('IO_CANCELLED', 'llm') }, 'IO_CANCELLED'],
      ['deadline', async () => { throw new IOControlError('IO_DEADLINE_EXCEEDED', 'llm') }, 'IO_DEADLINE_EXCEEDED'],
      ['generic', async () => { throw Object.assign(new Error('secret-stack'), { token: 'sk-abc' }) }, 'LLM_INVOCATION_FAILED'],
    ] as const) {
      const store = new MemoryEventStore()
      const port = new RecordingIOPort(new DefaultIOPort(new ScriptGateway(thrower)), store, `run-${label}`)
      const err = await port.invokeLLM(REQ).then(
        () => { throw new Error('expected reject') },
        e => e,
      )
      if (code === 'LLM_INVOCATION_FAILED') expect(err).toBeInstanceOf(LlmInvocationError)
      else expect(err).toBeInstanceOf(IOControlError)
      const term = (await store.readByRunId(`run-${label}`)).find(e => e.type === 'llm.responded')!
      expect(term.payload).toMatchObject({ status: 'error', error: { code } })
      expect(JSON.stringify(term.payload)).not.toContain('secret')
      expect(JSON.stringify(term.payload)).not.toContain('sk-abc')
    }
  })

  it('rejects invalid control before request append and provider call', async () => {
    const store = new MemoryEventStore()
    const gateway = new ScriptGateway(async () => OK)
    const port = new RecordingIOPort(new DefaultIOPort(gateway), store, 'run-v')
    await expect(port.invokeLLM(REQ, { control: { deadlineAt: Number.NaN } }))
      .rejects.toBeInstanceOf(IOInvocationValidationError)
    expect(await store.readByRunId('run-v')).toEqual([])
    expect(gateway.calls).toBe(0)
  })

  it('request pre-commit rejection: TraceWriteError stage=request, provider 0', async () => {
    const store = new CountingStore()
    store.rejectOnCall = 1
    const gateway = new ScriptGateway(async () => OK)
    const port = new RecordingIOPort(new DefaultIOPort(gateway), store, 'run-pre')
    await expect(port.invokeLLM(REQ)).rejects.toMatchObject({
      name: 'TraceWriteError',
      stage: 'request',
      operation: 'llm',
    })
    expect(gateway.calls).toBe(0)
    expect(store.events).toHaveLength(0)
  })

  it('terminal pre-commit rejection: request 1, terminal 0, provider 1, no second append', async () => {
    const store = new CountingStore()
    store.rejectOnCall = 2
    const gateway = new ScriptGateway(async () => OK)
    const port = new RecordingIOPort(new DefaultIOPort(gateway), store, 'run-term-pre')
    await expect(port.invokeLLM(REQ)).rejects.toMatchObject({ name: 'TraceWriteError', stage: 'terminal' })
    expect(gateway.calls).toBe(1)
    expect(store.events.map(e => e.type)).toEqual(['llm.requested'])
    expect(store.appendCalls).toBe(2)
  })

  it('terminal pre-commit rejection details', async () => {
    const store = new CountingStore()
    store.rejectOnCall = 2
    const gateway = new ScriptGateway(async () => {
      throw new ModelGatewayError({
        code: 'MODEL_TIMEOUT',
        message: SAFE_MESSAGES.MODEL_TIMEOUT,
        phase: 'request',
        provider: 'x',
        model: 'm',
        retryable: true,
      })
    })
    const port = new RecordingIOPort(new DefaultIOPort(gateway), store, 'run-term-pre2')
    const err = await port.invokeLLM(REQ).then(() => null, e => e)
    expect(err).toBeInstanceOf(TraceWriteError)
    expect(err.stage).toBe('terminal')
    expect(gateway.calls).toBe(1)
    expect(store.events.map(e => e.type)).toEqual(['llm.requested'])
    expect(store.appendCalls).toBe(2)
  })

  it('terminal after-commit rejection: event exists, caller still gets TraceWriteError', async () => {
    const store = new CountingStore()
    store.rejectAfterCommitOnCall = 2
    const gateway = new ScriptGateway(async () => OK)
    const port = new RecordingIOPort(new DefaultIOPort(gateway), store, 'run-after')
    await expect(port.invokeLLM(REQ)).rejects.toMatchObject({ stage: 'terminal' })
    expect(store.events.map(e => e.type)).toEqual(['llm.requested', 'llm.responded'])
    // CacheIndex builds from facts
    const idx = CacheIndex.fromEvents(store.events)
    expect(idx.consumeLLM(hashModelRequest(REQ)).content[0]).toMatchObject({ text: 'ok' })
  })

  it('success updates lastLlmRespondedId and returns response', async () => {
    const store = new MemoryEventStore()
    const cursor = new CausalCursor()
    const port = new RecordingIOPort(
      new DefaultIOPort(new ScriptGateway(async () => OK)),
      store,
      'run-ok',
      'runtime',
      undefined,
      cursor,
    )
    await expect(port.invokeLLM(REQ)).resolves.toEqual(OK)
    const term = (await store.readByRunId('run-ok')).find(e => e.type === 'llm.responded')!
    expect(term.payload).toMatchObject({ status: 'ok' })
    expect(cursor.lastLlmRespondedId).toBe(term.id)
    expect(cursor.lastLlmTerminalId).toBe(term.id)
  })
})

describe('ReplayingIOPort LLM failure reconstruct', () => {
  it('replays interleaved success/error with 0 provider calls', async () => {
    const store = new MemoryEventStore()
    let n = 0
    const liveGw = new ScriptGateway(async () => {
      n++
      if (n === 1) return OK
      if (n === 2) {
        throw new ModelGatewayError({
          code: 'MODEL_TIMEOUT',
          message: 'raw',
          phase: 'request',
          provider: 'p',
          model: 'm',
          retryable: true,
        })
      }
      throw new Error('secret-generic')
    })
    const recording = new RecordingIOPort(
      new DefaultIOPort(liveGw),
      store,
      'run-r',
      'runtime',
      undefined,
      undefined,
      'openai-compatible',
    )
    await expect(recording.invokeLLM(REQ)).resolves.toEqual(OK)
    await expect(recording.invokeLLM(REQ)).rejects.toBeInstanceOf(ModelGatewayError)
    await expect(recording.invokeLLM(REQ)).rejects.toBeInstanceOf(LlmInvocationError)

    const events = await store.readByRunId('run-r')
    const cache = CacheIndex.fromEvents(events)
    let providerCalls = 0
    const neverGw: IModelGateway = {
      async complete() { providerCalls++; throw new Error('provider must not be called') },
      async *stream() { providerCalls++; yield* [] },
    }
    const replay = new ReplayingIOPort(cache, new DefaultIOPort(neverGw))
    await expect(replay.invokeLLM(REQ)).resolves.toEqual(OK)
    await expect(replay.invokeLLM(REQ)).rejects.toMatchObject({
      name: 'ModelGatewayError',
      envelope: { code: 'MODEL_TIMEOUT', provider: 'openai-compatible' },
    })
    await expect(replay.invokeLLM(REQ)).rejects.toMatchObject({
      name: 'LlmInvocationError',
      envelope: { code: 'LLM_INVOCATION_FAILED' },
    })
    expect(providerCalls).toBe(0)
  })
})
