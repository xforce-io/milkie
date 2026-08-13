import { Milkie } from '../../src/runtime/Milkie'
import { MemoryStore } from '../../src/store/MemoryStore'
import { MemoryEventStore } from '../../src/trace/MemoryEventStore'
import { ModelGatewayError, SAFE_MESSAGES } from '../../src/gateway/ModelGatewayError'
import { runEventsToMessages } from '../../src/trace/diagnostics/sessionHistory'
import { buildExecutionProjection } from '../../src/trace/diagnostics/buildExecutionProjection'
import { buildDecisionSpine } from '../../src/trace/diagnostics/buildDecisionSpine'
import type { AgentConfig } from '../../src/types/agent'
import type {
  GatewayInvocationOptions,
  IModelGateway,
  ModelEvent,
  ModelRequest,
  ModelResponse,
} from '../../src/types/model'
import { IOControlError, LlmInvocationError } from '../../src/types/model'

const CFG: AgentConfig = {
  agentId: 'fail-llm',
  version: '1.0.0',
  systemPrompt: 'test',
  fsm: { states: [{ name: 'react', type: 'llm' }] },
  model: { provider: 'test', model: 'test-model', adapter: 'test' },
}

function deferred<T = void>() {
  let resolve!: (v: T) => void
  const promise = new Promise<T>(res => { resolve = res })
  return { promise, resolve }
}

class ScriptedGateway implements IModelGateway {
  calls = 0
  private i = 0
  constructor(private readonly scripts: Array<() => Promise<ModelResponse>>) {}
  async complete(_req: ModelRequest, _opts?: GatewayInvocationOptions): Promise<ModelResponse> {
    this.calls++
    const fn = this.scripts[this.i++]
    if (!fn) throw new Error('script exhausted')
    return fn()
  }
  async *stream(): AsyncIterable<ModelEvent> { yield* [] }
}

function assertNoSecrets(value: unknown) {
  const json = JSON.stringify(value)
  expect(json).not.toContain('SECRET')
  expect(json).not.toContain('sk-live')
  expect(json).not.toContain('raw-stack')
}

describe('S1/S2: LLM failure terminals and deterministic replay (#229)', () => {
  it('S1: rate-limit / timeout / cancel / deadline / generic each get one error terminal', async () => {
    const cases: Array<{
      name: string
      script: () => Promise<ModelResponse>
      code: string
      errClass: new (...args: never[]) => Error
    }> = [
      {
        name: 'rate-limit',
        script: async () => {
          throw new ModelGatewayError({
            code: 'MODEL_RATE_LIMITED',
            message: 'SECRET rate body sk-live',
            phase: 'request',
            provider: 'evil',
            model: 'evil',
            retryable: true,
            status: 429,
          })
        },
        code: 'MODEL_RATE_LIMITED',
        errClass: ModelGatewayError,
      },
      {
        name: 'timeout',
        script: async () => {
          throw new ModelGatewayError({
            code: 'MODEL_TIMEOUT',
            message: 'SECRET timeout raw-stack',
            phase: 'stream_read',
            provider: 'evil',
            model: 'evil',
            retryable: true,
          })
        },
        code: 'MODEL_TIMEOUT',
        errClass: ModelGatewayError,
      },
      {
        name: 'cancel',
        script: async () => { throw new IOControlError('IO_CANCELLED', 'llm') },
        code: 'IO_CANCELLED',
        errClass: IOControlError,
      },
      {
        name: 'deadline',
        script: async () => { throw new IOControlError('IO_DEADLINE_EXCEEDED', 'llm') },
        code: 'IO_DEADLINE_EXCEEDED',
        errClass: IOControlError,
      },
      {
        name: 'generic',
        script: async () => { throw Object.assign(new Error('SECRET raw-stack'), { token: 'sk-live' }) },
        code: 'LLM_INVOCATION_FAILED',
        errClass: LlmInvocationError,
      },
    ]

    for (const c of cases) {
      const gateway = new ScriptedGateway([c.script])
      const eventStore = new MemoryEventStore()
      const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore, gateway })
      milkie.registerAgent(CFG)

      const result = await milkie.invoke({
        agentId: CFG.agentId,
        goal: c.name,
        input: c.name,
      })

      if (c.code === 'IO_CANCELLED' || c.code === 'IO_DEADLINE_EXCEEDED') {
        expect(result.status).not.toBe('error')
        expect(result.stopCode).toBe(c.code)
        expect(result.stopReason).toBe(c.code === 'IO_CANCELLED' ? 'cancelled' : 'deadline')
      } else {
        expect(result.status).toBe('error')
        expect(result.error?.code).toBe(c.code)
        assertNoSecrets(result.error)
      }

      const events = await eventStore.readByRunId(result.agentRunId)
      const reqs = events.filter(e => e.type === 'llm.requested')
      const terms = events.filter(e => e.type === 'llm.responded')
      expect(reqs).toHaveLength(1)
      expect(terms).toHaveLength(1)
      expect(reqs[0]!.payload).toMatchObject({ outcomeSchemaVersion: 2 })
      expect(terms[0]!.causedBy).toBe(reqs[0]!.id)
      expect(terms[0]!.payload).toMatchObject({
        status: 'error',
        error: { code: c.code },
      })
      if (result.status === 'error') {
        expect(result.error).toEqual((terms[0]!.payload as { error: unknown }).error)
      }
      assertNoSecrets(terms[0]!.payload)

      // consumers
      const history = runEventsToMessages(events)
      expect(history.every(m => m.role !== 'assistant')).toBe(true)
      const step = buildExecutionProjection(events).steps.find(s => s.kind === 'llm')
      expect(step?.status).toBe('error')
      expect(step?.label).toBe(`LLM failure · ${c.code}`)
      expect(step?.response).toBeUndefined()
      const spine = buildDecisionSpine(events).nodes.find(n => n.kind === 'llm')
      expect(spine?.label).toBe(`LLM failure · ${c.code}`)
      expect(spine?.label).not.toBe('LLM → 文本')
    }
  })

  it('S2: interleaved success/model/control/generic replay with providerCalls=0', async () => {
    const ok: ModelResponse = {
      content: [{ type: 'text', text: 'hello' }],
      toolCalls: [],
      finishReason: 'end_turn',
    }
    const gateway = new ScriptedGateway([
      async () => ok,
      async () => {
        throw new ModelGatewayError({
          code: 'MODEL_AUTH_ERROR',
          message: 'SECRET auth',
          phase: 'request',
          provider: 'x',
          model: 'y',
          retryable: false,
        })
      },
      async () => { throw new IOControlError('IO_CANCELLED', 'llm') },
      async () => { throw new Error('SECRET generic') },
    ])
    // Force four separate single-turn runs (same request shape → same hash)
    // then stitch is not needed: one run with multi-turn would need tools.
    // Instead, record four independent runs and replay each.
    const live: Array<{ runId: string; error?: unknown; status: string }> = []
    const eventStore = new MemoryEventStore()

    for (let i = 0; i < 4; i++) {
      // fresh milkie shares eventStore+gateway script progression
      const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore, gateway })
      milkie.registerAgent(CFG)
      const result = await milkie.invoke({ agentId: CFG.agentId, goal: 'g', input: 'same' })
      live.push({ runId: result.agentRunId, error: result.error, status: result.status })
    }
    expect(gateway.calls).toBe(4)
    expect(live.map(l => l.status)).toEqual(['completed', 'error', 'interrupted', 'error'])
    expect(live[1]!.error).toMatchObject({ code: 'MODEL_AUTH_ERROR' })
    expect(live[2]!.error).toBeUndefined()
    expect(live[3]!.error).toMatchObject({ code: 'LLM_INVOCATION_FAILED' })

    // Replay each run with a counting gateway that must never be called.
    let replayProviderCalls = 0
    const neverGateway: IModelGateway = {
      async complete() { replayProviderCalls++; throw new Error('no provider') },
      async *stream() { replayProviderCalls++; yield* [] },
    }
    const replayMilkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway: neverGateway,
    })
    replayMilkie.registerAgent(CFG)

    const r0 = await replayMilkie.replay(live[0]!.runId)
    expect(r0.status).toBe('completed')
    expect(r0.output).toBe('hello')

    const r1 = await replayMilkie.replay(live[1]!.runId)
    expect(r1.status).toBe('error')
    expect(r1.error).toEqual(live[1]!.error)

    const r2 = await replayMilkie.replay(live[2]!.runId)
    expect(r2.status).toBe('interrupted')
    expect(r2.stopReason).toBe('cancelled')
    expect(r2.stopCode).toBe('IO_CANCELLED')

    const r3 = await replayMilkie.replay(live[3]!.runId)
    expect(r3.status).toBe('error')
    expect(r3.error).toEqual(live[3]!.error)

    expect(replayProviderCalls).toBe(0)
  })

  it('in-flight cancel still produces a control terminal when gateway observes abort', async () => {
    const entered = deferred()
    let signal: AbortSignal | undefined
    const gateway: IModelGateway = {
      async complete(_req, opts) {
        signal = opts?.signal
        entered.resolve()
        return new Promise<ModelResponse>((_res, rej) => {
          opts?.signal?.addEventListener('abort', () => {
            rej(new IOControlError('IO_CANCELLED', 'llm'))
          }, { once: true })
        })
      },
      async *stream() { yield* [] },
    }
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore, gateway })
    milkie.registerAgent(CFG)
    const controller = new AbortController()
    const invocation = milkie.invoke({
      agentId: CFG.agentId,
      goal: 'g',
      input: 'wait',
      control: { signal: controller.signal },
    })
    await entered.promise
    controller.abort()
    const result = await invocation
    expect(result.stopCode).toBe('IO_CANCELLED')
    expect(result.stopReason).toBe('cancelled')
    const events = await eventStore.readByRunId(result.agentRunId)
    const terms = events.filter(e => e.type === 'llm.responded')
    expect(terms).toHaveLength(1)
    expect(terms[0]!.payload).toMatchObject({
      status: 'error',
      error: { code: 'IO_CANCELLED', message: 'I/O invocation was cancelled.' },
    })
    expect(signal?.aborted).toBe(true)
  })
})
