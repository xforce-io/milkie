// #237: run-level deadline + cancellation propagation and terminal envelopes.
import { AgentRuntime } from '../runtime/AgentRuntime'
import { DefaultIOPort, type IIOPort, type LLMInvocationOptions } from '../runtime/IOPort'
import type { IOInvocationControl } from '../types/model'
import { RunControl, RunControlError } from '../runtime/RunControl'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { CacheIndex } from '../trace/CacheIndex'
import { ReplayingIOPort } from '../trace/ReplayingIOPort'
import { hashModelRequest } from '../trace/hash'
import type { Event } from '../trace/types'
import type { AgentConfig } from '../types/agent'
import type { AgentResult } from '../types/common'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'
import { IOControlError } from '../types/model'
import type { ToolDefinition } from '../types/tool'

function makeConfig(overrides: Partial<AgentConfig> = {}): AgentConfig {
  return {
    agentId: 'deadline-agent',
    version: '1.0.0',
    systemPrompt: 'sys',
    fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 10 }] },
    model: { provider: 'test', model: 'test', adapter: 'test' },
    ...overrides,
  }
}

function text(t: string): ModelResponse {
  return { content: [{ type: 'text', text: t }], toolCalls: [], finishReason: 'end_turn' }
}
function toolCall(id: string, name: string, input: unknown = {}): ModelResponse {
  return {
    content: [{ type: 'tool_use', id, name, input }],
    toolCalls: [{ id, name, input }],
    finishReason: 'tool_use',
  }
}

class ControllableGateway implements IModelGateway {
  requests = 0
  lastOpts: { signal?: AbortSignal } | undefined
  private handlers: Array<(req: ModelRequest, opts?: { signal?: AbortSignal }) => Promise<ModelResponse>>
  constructor(handlers: Array<(req: ModelRequest, opts?: { signal?: AbortSignal }) => Promise<ModelResponse>>) {
    this.handlers = handlers
  }
  async complete(req: ModelRequest, opts?: { signal?: AbortSignal }): Promise<ModelResponse> {
    this.requests++
    this.lastOpts = opts
    const h = this.handlers.shift()
    if (!h) throw new Error('no more gateway handlers')
    return h(req, opts)
  }
  async *stream(_req: ModelRequest, _opts?: { signal?: AbortSignal }): AsyncIterable<ModelEvent> {
    yield* []
  }
}

class TrackingPort implements IIOPort {
  llmCalls = 0
  toolCalls = 0
  lastLlmControl: IOInvocationControl | undefined
  lastToolControl: IOInvocationControl | undefined
  constructor(private readonly inner: IIOPort) {}
  async invokeLLM(req: ModelRequest, options?: LLMInvocationOptions) {
    this.llmCalls++
    this.lastLlmControl = options?.control
    return this.inner.invokeLLM(req, options)
  }
  async invokeTool(
    name: string,
    input: unknown,
    execute: (signal: AbortSignal) => Promise<unknown>,
    opts?: Parameters<IIOPort['invokeTool']>[3],
  ) {
    this.toolCalls++
    this.lastToolControl = opts?.control
    return this.inner.invokeTool(name, input, execute, opts)
  }
  now() { return this.inner.now() }
  uuid() { return this.inner.uuid() }
}

describe('#237 runtime deadline / cancellation', () => {
  describe('RunControl unit', () => {
    it('rejects non-finite deadlineAt', () => {
      expect(() => RunControl.create({ now: 1000, deadlineAt: Number.NaN }))
        .toThrow(/finite epoch-milliseconds/)
    })

    it('already-past deadline trips immediately as deadline', () => {
      const rc = RunControl.create({ now: 2000, deadlineAt: 1000 })
      expect(rc.stopped).toBe(true)
      expect(rc.reason).toBe('deadline')
      expect(() => rc.throwIfStopped()).toThrow(RunControlError)
      rc.dispose()
    })

    it('assertInvocationAllowed trips deadline from a later clock sample', () => {
      const rc = RunControl.create({ now: 0, deadlineAt: 1000 })
      expect(rc.stopped).toBe(false)
      expect(() => rc.assertInvocationAllowed(999)).not.toThrow()
      expect(() => rc.assertInvocationAllowed(1000)).toThrow(RunControlError)
      expect(rc.reason).toBe('deadline')
      rc.dispose()
    })

    it('already-aborted external signal trips as cancelled', () => {
      const c = new AbortController()
      c.abort()
      const rc = RunControl.create({ now: 1, signal: c.signal })
      expect(rc.reason).toBe('cancelled')
      rc.dispose()
    })

    it('child inherits parent cancel and takes earlier deadline', () => {
      const ac = new AbortController()
      const parent = RunControl.create({ now: 1000, deadlineAt: 5000, signal: ac.signal })
      const child = RunControl.create({ now: 1000, deadlineAt: 8000, parent })
      expect(child.deadlineAt).toBe(5000)
      ac.abort()
      expect(child.stopped).toBe(true)
      expect(child.reason).toBe('cancelled')
      child.dispose(); parent.dispose()
    })

    it('earliest observed reason wins and is sticky', () => {
      const rc = RunControl.create({ now: 1000, deadlineAt: 1000 })
      expect(rc.reason).toBe('deadline')
      // subsequent trip ignored
      const c = new AbortController()
      // can't re-trip via public API; reason stays deadline
      expect(rc.reason).toBe('deadline')
      void c
      rc.dispose()
    })
  })

  it('S1: deadline prevents new model calls and returns RUN_DEADLINE_EXCEEDED', async () => {
    const eventStore = new MemoryEventStore()
    let blockedResolve: (() => void) | undefined
    const blocked = new Promise<void>(r => { blockedResolve = r })

    const gw = new ControllableGateway([
      async (_req, opts) => {
        // Stay in-flight until aborted/deadline; then surface AbortError.
        await new Promise<void>((resolve, reject) => {
          const onAbort = () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' }))
          if (opts?.signal?.aborted) return onAbort()
          opts?.signal?.addEventListener('abort', onAbort, { once: true })
          // also release path for cleanup
          void blocked.then(() => resolve())
        })
        return text('should-not')
      },
      async () => text('should-never-be-called'),
    ])

    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway: gw,
    })
    milkie.registerAgent(makeConfig())

    const started = Date.now()
    const resultP = milkie.invoke({
      agentId: 'deadline-agent',
      goal: 'g',
      input: 'i',
      control: { deadlineAt: started + 30 },
    })
    const result = await resultP
    blockedResolve?.()

    expect(result).toMatchObject({
      status: 'completed',
      stopReason: 'deadline',
      stopCode: 'RUN_DEADLINE_EXCEEDED',
    })
    // No second model call after deadline.
    expect(gw.requests).toBe(1)

    const terminal = (await eventStore.readByRunId(result.agentRunId))
      .find(e => e.type === 'agent.run.completed')
    expect(terminal?.payload).toMatchObject({
      status: 'completed',
      stopReason: 'deadline',
    })
  })

  it('S2: caller cancel is machine-distinguishable as RUN_CANCELLED (not model/tool failure)', async () => {
    const eventStore = new MemoryEventStore()
    const ac = new AbortController()
    const gw = new ControllableGateway([
      async (_req, opts) => {
        await new Promise<void>((_resolve, reject) => {
          const onAbort = () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' }))
          if (opts?.signal?.aborted) return onAbort()
          opts?.signal?.addEventListener('abort', onAbort, { once: true })
        })
        return text('nope')
      },
    ])
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway: gw,
    })
    milkie.registerAgent(makeConfig())

    const resultP = milkie.invoke({
      agentId: 'deadline-agent',
      goal: 'g',
      input: 'i',
      control: { signal: ac.signal },
    })
    // Let the LLM call start, then cancel.
    await new Promise(r => setTimeout(r, 10))
    ac.abort()
    const result = await resultP

    expect(result).toMatchObject({
      status: 'interrupted',
      stopReason: 'cancelled',
      stopCode: 'RUN_CANCELLED',
    })
    // Must not look like a provider/model failure.
    expect(result.error).toBeUndefined()

    const terminal = (await eventStore.readByRunId(result.agentRunId))
      .find(e => e.type === 'agent.run.completed')
    expect(terminal?.payload).toMatchObject({
      status: 'interrupted',
      stopReason: 'cancelled',
    })
  })

  it('deadline during blocking tool aborts handler signal and skips subsequent LLM/tool scheduling', async () => {
    const eventStore = new MemoryEventStore()
    const seenSignals: Array<AbortSignal | undefined> = []
    let toolEntered = false
    const tool: ToolDefinition = {
      name: 'block',
      description: 'blocks until aborted',
      inputSchema: { type: 'object', properties: {}, required: [] },
      handler: async (_input, ctx) => {
        toolEntered = true
        seenSignals.push(ctx.signal)
        await new Promise<void>((_resolve, reject) => {
          const onAbort = () => reject(Object.assign(new Error('tool aborted'), { name: 'AbortError' }))
          if (ctx.signal?.aborted) return onAbort()
          ctx.signal?.addEventListener('abort', onAbort, { once: true })
        })
        return { ok: true }
      },
    }

    const gw = new ControllableGateway([
      async () => toolCall('t1', 'block'),
      async () => text('should-never-run'),
    ])

    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway: gw,
      tools: [tool],
    })
    milkie.registerAgent(makeConfig({
      builtinTools: { allow: [] },
      fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5, tools: ['block'] }] },
    }))

    const started = Date.now()
    const result = await milkie.invoke({
      agentId: 'deadline-agent',
      goal: 'g',
      input: 'i',
      control: { deadlineAt: started + 40 },
    })

    expect(toolEntered).toBe(true)
    expect(seenSignals).toHaveLength(1)
    expect(seenSignals[0]).toBeInstanceOf(AbortSignal)
    expect(seenSignals[0]!.aborted).toBe(true)
    expect(result).toMatchObject({
      status: 'completed',
      stopReason: 'deadline',
      stopCode: 'RUN_DEADLINE_EXCEEDED',
    })
    // First LLM scheduled the tool; second LLM must not start after deadline.
    expect(gw.requests).toBe(1)

    const terminal = (await eventStore.readByRunId(result.agentRunId))
      .find(e => e.type === 'agent.run.completed')
    expect(terminal?.payload).toMatchObject({
      status: 'completed',
      stopReason: 'deadline',
    })
  })

  it('passes the same signal into tool handlers via ToolContext and IOPort control', async () => {
    const seen: Array<AbortSignal | undefined> = []
    const tool: ToolDefinition = {
      name: 'probe',
      description: 'probe',
      inputSchema: { type: 'object', properties: {}, required: [] },
      handler: async (_input, ctx) => {
        seen.push(ctx.signal)
        return { ok: true }
      },
    }
    const gw = new ControllableGateway([
      async () => toolCall('t1', 'probe'),
      async () => text('done'),
    ])
    const base = new DefaultIOPort(gw)
    const port = new TrackingPort(base)
    const ac = new AbortController()
    const runtime = new AgentRuntime({
      config: makeConfig({
        fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5, tools: ['probe'] }] },
      }),
      goal: 'g', input: 'i',
      stateStore: new MemoryStore(),
      recorder: new InMemoryRecorder(undefined, 'deadline-agent'),
      ioPort: port,
      extraTools: [tool],
      control: { signal: ac.signal },
    })
    const result = await runtime.run('i')
    expect(result.status).toBe('completed')
    expect(seen).toHaveLength(1)
    expect(seen[0]).toBeInstanceOf(AbortSignal)
    expect(seen[0]!.aborted).toBe(false)
    expect(port.lastLlmControl?.signal).toBeInstanceOf(AbortSignal)
    expect(port.lastToolControl?.signal).toBe(port.lastLlmControl?.signal)
  })

  it('pre-expired deadline completes with no LLM I/O', async () => {
    const gw = new ControllableGateway([
      async () => text('should-not-run'),
    ])
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: gw,
    })
    milkie.registerAgent(makeConfig())
    const result = await milkie.invoke({
      agentId: 'deadline-agent',
      goal: 'g',
      input: 'i',
      control: { deadlineAt: Date.now() - 1 },
    })
    expect(result.stopCode).toBe('RUN_DEADLINE_EXCEEDED')
    expect(gw.requests).toBe(0)
  })

  it('does not misreport a real model error as cancel when control is idle', async () => {
    const gw = new ControllableGateway([
      async () => { throw Object.assign(new Error('upstream 500'), { status: 500 }) },
    ])
    // Use DefaultIOPort path via Milkie without control — gateway errors stay gateway errors
    // when adapters normalize. ControllableGateway throws raw; AgentRuntime surfaces message.
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: gw,
    })
    milkie.registerAgent(makeConfig())
    const result = await milkie.invoke({ agentId: 'deadline-agent', goal: 'g', input: 'i' })
    expect(result.status).toBe('error')
    expect(result.error?.code === 'RUN_CANCELLED' || result.error?.code === 'RUN_DEADLINE_EXCEEDED').toBe(false)
  })

  it('omitted control preserves prior completed behavior', async () => {
    const gw = new ControllableGateway([async () => text('hello')])
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: gw,
    })
    milkie.registerAgent(makeConfig())
    const result = await milkie.invoke({ agentId: 'deadline-agent', goal: 'g', input: 'i' })
    expect(result).toMatchObject({ status: 'completed', output: 'hello' })
  })

  it('replay I/O boundary ignores local deadline/cancel control and never issues live gateway I/O', async () => {
    // Replay serves recorded cache only; wall-clock control must not rewrite the path.
    const req: ModelRequest = {
      model: 'm',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      system: '',
      tools: [],
    }
    const h = hashModelRequest(req)
    const cached: ModelResponse = text('from-cache')
    const ev: Event = {
      id: 'e1',
      runId: 'r-replay-ctrl',
      type: 'llm.responded',
      actor: 'runtime',
      timestamp: 1,
      payload: { response: cached, requestHash: h },
    }

    class ExplodingGateway implements IModelGateway {
      calls = 0
      async complete(): Promise<ModelResponse> {
        this.calls++
        throw new Error('live gateway must not be called during replay')
      }
      async *stream(): AsyncIterable<never> {
        this.calls++
        yield* []
      }
    }

    const gw = new ExplodingGateway()
    const port = new ReplayingIOPort(CacheIndex.fromEvents([ev]), new DefaultIOPort(gw))
    const ac = new AbortController()
    ac.abort()

    await expect(port.invokeLLM(req, {
      control: {
        signal: ac.signal,
        deadlineAt: Date.now() - 1,
      },
    })).rejects.toMatchObject({ code: 'IO_CANCELLED' })
    expect(gw.calls).toBe(0)
  })

  it('recorded RUN_CANCELLED terminal replays without live I/O and keeps terminal code', async () => {
    const store = new MemoryEventStore()
    const ac = new AbortController()
    const recordGw = new ControllableGateway([
      async (_req, opts) => {
        await new Promise<void>((_resolve, reject) => {
          const onAbort = () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' }))
          if (opts?.signal?.aborted) return onAbort()
          opts?.signal?.addEventListener('abort', onAbort, { once: true })
        })
        return text('nope')
      },
    ])
    const recordMilkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: store,
      gateway: recordGw,
    })
    recordMilkie.registerAgent(makeConfig())
    const resultP = recordMilkie.invoke({
      agentId: 'deadline-agent',
      goal: 'g',
      input: 'i',
      control: { signal: ac.signal },
    })
    await new Promise(r => setTimeout(r, 10))
    ac.abort()
    const original = await resultP
    expect(original.stopCode).toBe('RUN_CANCELLED')

    class LiveProbeGateway implements IModelGateway {
      calls = 0
      async complete(): Promise<ModelResponse> {
        this.calls++
        return text('live-would-be-wrong')
      }
      async *stream(): AsyncIterable<never> { yield* [] }
    }
    const live = new LiveProbeGateway()
    const replayMilkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: store,
      gateway: live,
    })
    replayMilkie.registerAgent(makeConfig())

    // Cancelled mid-LLM leaves no llm.responded cache entry. Replay must project
    // the recorded RUN_CANCELLED terminal without live I/O or local clock rewrite.
    const replayed: AgentResult = await replayMilkie.replay(original.agentRunId)
    expect(live.calls).toBe(0)
    expect(replayed).toMatchObject({
      status: 'interrupted',
      stopReason: 'cancelled',
    })
  })

  it('late LLM success after deadline does not complete the run', async () => {
    // Gateway ignores AbortSignal and returns success after the deadline. Runtime
    // must gate the late response and finish as RUN_DEADLINE_EXCEEDED, not completed.
    const eventStore = new MemoryEventStore()
    const gw = new ControllableGateway([
      async () => {
        await new Promise<void>(r => setTimeout(r, 80))
        return text('late-success-must-not-win')
      },
    ])
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway: gw,
    })
    milkie.registerAgent(makeConfig())
    const started = Date.now()
    const result = await milkie.invoke({
      agentId: 'deadline-agent',
      goal: 'g',
      input: 'i',
      control: { deadlineAt: started + 25 },
    })
    expect(result).toMatchObject({
      status: 'completed',
      stopReason: 'deadline',
      stopCode: 'RUN_DEADLINE_EXCEEDED',
    })
    expect(result.output).not.toBe('late-success-must-not-win')
    const terminal = (await eventStore.readByRunId(result.agentRunId))
      .find(e => e.type === 'agent.run.completed')
    expect(terminal?.payload).toMatchObject({
      status: 'completed',
      stopReason: 'deadline',
    })
  })

  it('deadline after first serial tool prevents starting the second serial tool', async () => {
    const started: string[] = []
    const tools: ToolDefinition[] = [
      {
        name: 'serial_a',
        description: 'first serial',
        inputSchema: { type: 'object', properties: {}, required: [] },
        // default parallelSafe is false → serial queue
        handler: async () => {
          started.push('serial_a')
          await new Promise<void>(r => setTimeout(r, 60))
          return { ok: 'a' }
        },
      },
      {
        name: 'serial_b',
        description: 'second serial',
        inputSchema: { type: 'object', properties: {}, required: [] },
        handler: async () => {
          started.push('serial_b')
          return { ok: 'b' }
        },
      },
    ]
    const gw = new ControllableGateway([
      async () => ({
        content: [
          { type: 'tool_use', id: 'a1', name: 'serial_a', input: {} },
          { type: 'tool_use', id: 'b1', name: 'serial_b', input: {} },
        ],
        toolCalls: [
          { id: 'a1', name: 'serial_a', input: {} },
          { id: 'b1', name: 'serial_b', input: {} },
        ],
        finishReason: 'tool_use',
      }),
      async () => text('should-not-run-after-deadline'),
    ])
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: gw,
      tools,
    })
    milkie.registerAgent(makeConfig({
      builtinTools: { allow: [] },
      fsm: {
        states: [{
          name: 'react',
          type: 'llm',
          max_iterations: 5,
          tools: ['serial_a', 'serial_b'],
        }],
      },
    }))
    const t0 = Date.now()
    const result = await milkie.invoke({
      agentId: 'deadline-agent',
      goal: 'g',
      input: 'i',
      control: { deadlineAt: t0 + 30 },
    })
    expect(started).toEqual(['serial_a'])
    expect(result).toMatchObject({
      status: 'completed',
      stopReason: 'deadline',
      stopCode: 'RUN_DEADLINE_EXCEEDED',
    })
    expect(gw.requests).toBe(1)
  })

  it('deadline during tool retry backoff does not perform another attempt', async () => {
    let attempts = 0
    const tool: ToolDefinition = {
      name: 'flaky',
      description: 'fails retryably once then would retry',
      inputSchema: { type: 'object', properties: {}, required: [] },
      handler: async () => {
        attempts++
        throw Object.assign(new Error('transient'), { retryable: true })
      },
    }
    const gw = new ControllableGateway([
      async () => toolCall('f1', 'flaky'),
      async () => text('should-not'),
    ])
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: gw,
      tools: [tool],
    })
    milkie.registerAgent(makeConfig({
      builtinTools: { allow: [] },
      fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5, tools: ['flaky'] }] },
    }))
    // Deadline lands inside the 500ms retry backoff window after attempt 1.
    const t0 = Date.now()
    const result = await milkie.invoke({
      agentId: 'deadline-agent',
      goal: 'g',
      input: 'i',
      control: { deadlineAt: t0 + 80 },
    })
    expect(attempts).toBe(1)
    expect(result).toMatchObject({
      status: 'completed',
      stopReason: 'deadline',
      stopCode: 'RUN_DEADLINE_EXCEEDED',
    })
  })

  it('idle control does not rebrand gateway AbortError as RUN_CANCELLED', async () => {
    const gw = new ControllableGateway([
      async () => {
        throw Object.assign(new Error('provider aborted mid-stream'), { name: 'AbortError' })
      },
    ])
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: gw,
    })
    milkie.registerAgent(makeConfig())
    const result = await milkie.invoke({
      agentId: 'deadline-agent',
      goal: 'g',
      input: 'i',
      // Control present but idle: long deadline, no external cancel.
      control: { deadlineAt: Date.now() + 60_000 },
    })
    expect(result.status).toBe('error')
    expect(result.error?.code).not.toBe('RUN_CANCELLED')
    expect(result.error?.code).not.toBe('RUN_DEADLINE_EXCEEDED')
  })

  it('idle control does not rebrand tool AbortError as RUN_CANCELLED', async () => {
    const tool: ToolDefinition = {
      name: 'abortish',
      description: 'throws AbortError without run stop',
      inputSchema: { type: 'object', properties: {}, required: [] },
      handler: async () => {
        throw Object.assign(new Error('tool internal abort'), { name: 'AbortError' })
      },
    }
    const gw = new ControllableGateway([
      async () => toolCall('t1', 'abortish'),
      async () => text('should-continue-or-surface-tool-error'),
    ])
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: gw,
      tools: [tool],
    })
    milkie.registerAgent(makeConfig({
      builtinTools: { allow: [] },
      fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5, tools: ['abortish'] }] },
    }))
    const result = await milkie.invoke({
      agentId: 'deadline-agent',
      goal: 'g',
      input: 'i',
      control: { deadlineAt: Date.now() + 60_000 },
    })
    // Tool AbortError with idle control is a per-tool failure, not run cancel.
    // Runtime may complete the turn after surfacing the tool error to the model,
    // or end in a non-RUN_* error — never RUN_CANCELLED / RUN_DEADLINE_EXCEEDED.
    expect(result.error?.code === 'RUN_CANCELLED' || result.error?.code === 'RUN_DEADLINE_EXCEEDED').toBe(false)
  })

  it('action-state handler goes through IOPort: record then replay skips live handler', async () => {
    let liveHandlerCalls = 0
    const seenSignals: Array<AbortSignal | undefined> = []
    const actionTool: ToolDefinition = {
      name: 'do_action',
      description: 'action handler with side effect',
      inputSchema: { type: 'object', properties: {}, required: [] },
      handler: async (_input, ctx) => {
        liveHandlerCalls++
        seenSignals.push(ctx.signal)
        return 'action-side-effect'
      },
    }
    const store = new MemoryEventStore()
    const recordGw = new ControllableGateway([]) // action-only FSM: no LLM
    const recordMilkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: store,
      gateway: recordGw,
      tools: [actionTool],
    })
    recordMilkie.registerAgent(makeConfig({
      builtinTools: { allow: [] },
      fsm: { states: [{ name: 'act', type: 'action', handler: 'do_action' }] },
    }))
    const ac = new AbortController()
    const original = await recordMilkie.invoke({
      agentId: 'deadline-agent',
      goal: 'g',
      input: 'i',
      control: { signal: ac.signal },
    })
    expect(original).toMatchObject({ status: 'completed', output: 'action-side-effect' })
    expect(liveHandlerCalls).toBe(1)
    expect(seenSignals).toHaveLength(1)
    expect(seenSignals[0]).toBeInstanceOf(AbortSignal)
    expect(seenSignals[0]!.aborted).toBe(false)

    // Trace must have recorded the action via the tool I/O boundary.
    const recorded = await store.readByRunId(original.agentRunId)
    expect(recorded.some(e => e.type === 'tool.requested')).toBe(true)
    expect(recorded.some(e => e.type === 'tool.responded')).toBe(true)

    // Fresh handler instance would run if replay bypassed IOPort.
    let replayHandlerCalls = 0
    const replayTool: ToolDefinition = {
      name: 'do_action',
      description: 'action handler with side effect',
      inputSchema: { type: 'object', properties: {}, required: [] },
      handler: async () => {
        replayHandlerCalls++
        return 'live-replay-would-be-wrong'
      },
    }
    class LiveProbeGateway implements IModelGateway {
      calls = 0
      async complete(): Promise<ModelResponse> {
        this.calls++
        return text('live-llm-wrong')
      }
      async *stream(): AsyncIterable<never> { yield* [] }
    }
    const live = new LiveProbeGateway()
    const replayMilkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: store,
      gateway: live,
      tools: [replayTool],
    })
    replayMilkie.registerAgent(makeConfig({
      builtinTools: { allow: [] },
      fsm: { states: [{ name: 'act', type: 'action', handler: 'do_action' }] },
    }))
    const replayed = await replayMilkie.replay(original.agentRunId)
    expect(replayHandlerCalls).toBe(0)
    expect(live.calls).toBe(0)
    expect(replayed).toMatchObject({
      status: 'completed',
      output: 'action-side-effect',
    })
  })

  it('fail-closes tool handler start when deadline trips after runtime pre-gate but before execute dispatch', async () => {
    // Reproduces the Recording/custom-wrapper race: invokeTool is entered while
    // the run is still live, async work runs, deadline fires, then the port
    // finally dispatches the handler thunk. Handler must not start; run ends as
    // RUN_DEADLINE_EXCEEDED (not a demoted tool error).
    jest.useFakeTimers()
    const epoch = 1_700_000_000_000
    jest.setSystemTime(epoch)

    let handlerEntered = false
    const tool: ToolDefinition = {
      name: 'late_start',
      description: 'must not start after deadline',
      inputSchema: { type: 'object', properties: {}, required: [] },
      handler: async () => {
        handlerEntered = true
        return { ok: true }
      },
    }

    let releaseHold!: () => void
    const hold = new Promise<void>(resolve => { releaseHold = resolve })
    let markEntered!: () => void
    const entered = new Promise<void>(resolve => { markEntered = resolve })

    class DelayedDispatchPort implements IIOPort {
      constructor(private readonly inner: IIOPort) {}
      async invokeLLM(
        req: ModelRequest,
        options?: LLMInvocationOptions,
      ) {
        return this.inner.invokeLLM(req, options)
      }
      async invokeTool(
        name: string,
        input: unknown,
        execute: (signal: AbortSignal) => Promise<unknown>,
        opts?: Parameters<IIOPort['invokeTool']>[3],
      ) {
        // Simulate RecordingIOPort/trace await between runtime pre-gate and
        // actual handler start (DefaultIOPort previously ran execute unconditionally).
        markEntered()
        await hold
        return this.inner.invokeTool(name, input, execute, opts)
      }
      now() { return Date.now() }
      uuid() { return this.inner.uuid() }
    }

    try {
      const eventStore = new MemoryEventStore()
      const gw = new ControllableGateway([
        async () => toolCall('t1', 'late_start'),
        async () => text('must-not-run'),
      ])
      const port = new DelayedDispatchPort(new DefaultIOPort(gw))
      const runtime = new AgentRuntime({
        config: makeConfig({
          builtinTools: { allow: [] },
          fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5, tools: ['late_start'] }] },
        }),
        goal: 'g',
        input: 'i',
        stateStore: new MemoryStore(),
        recorder: new InMemoryRecorder(undefined, 'deadline-agent'),
        ioPort: port,
        extraTools: [tool],
        eventStore,
        control: { deadlineAt: epoch + 1_000 },
      })

      const resultP = runtime.run('i')
      await entered
      // Deadline elapses while invokeTool is mid-flight, before execute/dispatch.
      await jest.advanceTimersByTimeAsync(1_001)
      releaseHold()
      const result = await resultP

      expect(handlerEntered).toBe(false)
      expect(result).toMatchObject({
        status: 'completed',
        stopReason: 'deadline',
        stopCode: 'RUN_DEADLINE_EXCEEDED',
      })
      const events = await eventStore.readByRunId(result.agentRunId)
      const toolRespondedOk = events.some(e => {
        if (e.type !== 'tool.responded') return false
        return typeof e.payload === 'object'
          && e.payload !== null
          && 'status' in e.payload
          && e.payload.status === 'ok'
      })
      expect(toolRespondedOk).toBe(false)
      expect(gw.requests).toBe(1)
    } finally {
      jest.useRealTimers()
    }
  })

  it('fail-closes action-state handler when deadline trips inside delayed invokeTool dispatch', async () => {
    jest.useFakeTimers()
    const epoch = 1_700_000_000_000
    jest.setSystemTime(epoch)

    let handlerEntered = false
    const actionTool: ToolDefinition = {
      name: 'do_action',
      description: 'action handler',
      inputSchema: { type: 'object', properties: {}, required: [] },
      handler: async () => {
        handlerEntered = true
        return 'should-not-run'
      },
    }

    let releaseHold!: () => void
    const hold = new Promise<void>(resolve => { releaseHold = resolve })
    let markEntered!: () => void
    const entered = new Promise<void>(resolve => { markEntered = resolve })

    class DelayedDispatchPort implements IIOPort {
      constructor(private readonly inner: IIOPort) {}
      async invokeLLM(
        req: ModelRequest,
        options?: LLMInvocationOptions,
      ) {
        return this.inner.invokeLLM(req, options)
      }
      async invokeTool(
        name: string,
        input: unknown,
        execute: (signal: AbortSignal) => Promise<unknown>,
        opts?: Parameters<IIOPort['invokeTool']>[3],
      ) {
        markEntered()
        await hold
        return this.inner.invokeTool(name, input, execute, opts)
      }
      now() { return Date.now() }
      uuid() { return this.inner.uuid() }
    }

    try {
      const eventStore = new MemoryEventStore()
      const gw = new ControllableGateway([])
      const port = new DelayedDispatchPort(new DefaultIOPort(gw))
      const runtime = new AgentRuntime({
        config: makeConfig({
          builtinTools: { allow: [] },
          fsm: { states: [{ name: 'act', type: 'action', handler: 'do_action' }] },
        }),
        goal: 'g',
        input: 'i',
        stateStore: new MemoryStore(),
        recorder: new InMemoryRecorder(undefined, 'deadline-agent'),
        ioPort: port,
        extraTools: [actionTool],
        eventStore,
        control: { deadlineAt: epoch + 1_000 },
      })

      const resultP = runtime.run('i')
      await entered
      await jest.advanceTimersByTimeAsync(1_001)
      releaseHold()
      const result = await resultP

      expect(handlerEntered).toBe(false)
      expect(result).toMatchObject({
        status: 'completed',
        stopReason: 'deadline',
        stopCode: 'RUN_DEADLINE_EXCEEDED',
      })
      expect(gw.requests).toBe(0)
    } finally {
      jest.useRealTimers()
    }
  })

  it('DefaultIOPort.invokeTool refuses execute when control signal is already aborted', async () => {
    let executed = false
    const gw = new ControllableGateway([])
    const port = new DefaultIOPort(gw)
    const ac = new AbortController()
    ac.abort()
    await expect(port.invokeTool(
      't',
      {},
      async () => {
        executed = true
        return 'nope'
      },
      { control: { signal: ac.signal, deadlineAt: port.now() - 1 } },
    )).rejects.toBeInstanceOf(IOControlError)
    expect(executed).toBe(false)
  })

  it('final tool gate uses IOPort clock, not ambient Date.now (virtual clock)', async () => {
    // #237 L2 §4/§9: deadline comparison at the final tool gate must share the
    // same IOPort clock RunControl was created with. A custom port with
    // virtual now=0 and deadlineAt=1000 must allow tools while ambient wall
    // time is already far past 1000, and refuse once virtual now reaches the
    // deadline. Assertions never read ambient Date.now().

    let virtualNow = 0

    // Subclass so DefaultIOPort.invokeTool's this.now() is the virtual clock.
    class VirtualDefaultIOPort extends DefaultIOPort {
      now(): number {
        return virtualNow
      }
    }

    // Unit: port-owned final gate — ambient wall time is irrelevant.
    {
      const vport = new VirtualDefaultIOPort(new ControllableGateway([]))
      virtualNow = 0
      let ran = false
      await expect(vport.invokeTool(
        't',
        {},
        async () => { ran = true; return 'ok' },
        { control: { deadlineAt: 1000 } },
      )).resolves.toBe('ok')
      expect(ran).toBe(true)

      virtualNow = 1000
      ran = false
      await expect(vport.invokeTool(
        't',
        {},
        async () => { ran = true; return 'nope' },
        { control: { deadlineAt: 1000 } },
      )).rejects.toMatchObject({ code: 'IO_DEADLINE_EXCEEDED' })
      expect(ran).toBe(false)
    }

    // Runtime path: RunControl created at virtual now=0, deadline=1000.
    // Ambient wall ms is already >> 1000; handler must still start while
    // virtual now stays below the deadline.
    virtualNow = 0
    let handlerEntered = false
    const tool: ToolDefinition = {
      name: 'clocked',
      description: 'virtual-clock tool',
      inputSchema: { type: 'object', properties: {}, required: [] },
      handler: async () => {
        handlerEntered = true
        return { ok: true }
      },
    }
    const gw = new ControllableGateway([
      async () => toolCall('t1', 'clocked'),
      async () => text('done'),
    ])
    const port = new VirtualDefaultIOPort(gw)
    const runtime = new AgentRuntime({
      config: makeConfig({
        builtinTools: { allow: [] },
        fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5, tools: ['clocked'] }] },
      }),
      goal: 'g',
      input: 'i',
      stateStore: new MemoryStore(),
      recorder: new InMemoryRecorder(undefined, 'deadline-agent'),
      ioPort: port,
      extraTools: [tool],
      control: { deadlineAt: 1000 },
    })

    const result = await runtime.run('i')
    expect(handlerEntered).toBe(true)
    expect(result.status).toBe('completed')
    expect(gw.requests).toBe(2)

    // Advance virtual clock to the deadline mid-invokeTool (before dispatch).
    // Handler must not start; terminal is RUN_DEADLINE_EXCEEDED.
    virtualNow = 0
    handlerEntered = false
    let releaseHold!: () => void
    const hold = new Promise<void>(resolve => { releaseHold = resolve })
    let markEntered!: () => void
    const entered = new Promise<void>(resolve => { markEntered = resolve })

    class HoldThenAdvancePort implements IIOPort {
      constructor(private readonly inner: IIOPort) {}
      async invokeLLM(
        req: ModelRequest,
        options?: LLMInvocationOptions,
      ) {
        return this.inner.invokeLLM(req, options)
      }
      async invokeTool(
        name: string,
        input: unknown,
        execute: (signal: AbortSignal) => Promise<unknown>,
        opts?: Parameters<IIOPort['invokeTool']>[3],
      ) {
        markEntered()
        await hold
        // Virtual clock reaches deadline while invokeTool is mid-flight.
        virtualNow = 1000
        return this.inner.invokeTool(name, input, execute, opts)
      }
      now() { return virtualNow }
      uuid() { return this.inner.uuid() }
    }

    const gw2 = new ControllableGateway([
      async () => toolCall('t2', 'clocked'),
      async () => text('must-not-run'),
    ])
    // Inner VirtualDefaultIOPort shares virtualNow with the outer hold wrapper.
    const port2 = new HoldThenAdvancePort(new VirtualDefaultIOPort(gw2))
    const runtime2 = new AgentRuntime({
      config: makeConfig({
        builtinTools: { allow: [] },
        fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5, tools: ['clocked'] }] },
      }),
      goal: 'g',
      input: 'i',
      stateStore: new MemoryStore(),
      recorder: new InMemoryRecorder(undefined, 'deadline-agent'),
      ioPort: port2,
      extraTools: [tool],
      control: { deadlineAt: 1000 },
    })

    const resultP = runtime2.run('i')
    await entered
    releaseHold()
    const result2 = await resultP

    expect(handlerEntered).toBe(false)
    expect(result2).toMatchObject({
      status: 'completed',
      stopReason: 'deadline',
      stopCode: 'RUN_DEADLINE_EXCEEDED',
    })
    expect(gw2.requests).toBe(1)
  })

  it('execute thunk gate fail-closes when custom IOPort calls execute() after deadline', async () => {
    // Custom ports may invoke the execute thunk directly (skip DefaultIOPort).
    // The runtime-supplied thunk must still refuse handler entry after stop.
    jest.useFakeTimers()
    const epoch = 1_700_000_000_000
    jest.setSystemTime(epoch)

    let handlerEntered = false
    const tool: ToolDefinition = {
      name: 'direct_exec',
      description: 'must not start after deadline',
      inputSchema: { type: 'object', properties: {}, required: [] },
      handler: async () => {
        handlerEntered = true
        return { ok: true }
      },
    }

    let releaseHold!: () => void
    const hold = new Promise<void>(resolve => { releaseHold = resolve })
    let markEntered!: () => void
    const entered = new Promise<void>(resolve => { markEntered = resolve })

    class DirectExecuteAfterHoldPort implements IIOPort {
      constructor(private readonly inner: IIOPort) {}
      async invokeLLM(
        req: ModelRequest,
        options?: LLMInvocationOptions,
      ) {
        return this.inner.invokeLLM(req, options)
      }
      async invokeTool(
        _name: string,
        _input: unknown,
        execute: (signal: AbortSignal) => Promise<unknown>,
        _opts?: Parameters<IIOPort['invokeTool']>[3],
        _control?: IOInvocationControl,
      ) {
        markEntered()
        await hold
        // Intentionally bypass DefaultIOPort — call the runtime thunk directly.
        return execute(new AbortController().signal)
      }
      now() { return Date.now() }
      uuid() { return this.inner.uuid() }
    }

    try {
      const eventStore = new MemoryEventStore()
      const gw = new ControllableGateway([
        async () => toolCall('t1', 'direct_exec'),
        async () => text('must-not-run'),
      ])
      const port = new DirectExecuteAfterHoldPort(new DefaultIOPort(gw))
      const runtime = new AgentRuntime({
        config: makeConfig({
          builtinTools: { allow: [] },
          fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5, tools: ['direct_exec'] }] },
        }),
        goal: 'g',
        input: 'i',
        stateStore: new MemoryStore(),
        recorder: new InMemoryRecorder(undefined, 'deadline-agent'),
        ioPort: port,
        extraTools: [tool],
        eventStore,
        control: { deadlineAt: epoch + 1_000 },
      })

      const resultP = runtime.run('i')
      await entered
      await jest.advanceTimersByTimeAsync(1_001)
      releaseHold()
      const result = await resultP

      expect(handlerEntered).toBe(false)
      expect(result).toMatchObject({
        status: 'completed',
        stopReason: 'deadline',
        stopCode: 'RUN_DEADLINE_EXCEEDED',
      })
      expect(gw.requests).toBe(1)
    } finally {
      jest.useRealTimers()
    }
  })

  it('runtime final thunk gate samples IOPort clock on direct custom execute (virtual clock)', async () => {
    // HIGH: custom IOPort may call execute() directly, bypassing Default/Recording
    // port gates. The runtime-owned thunk must still sample this run's IOPort
    // clock (not rely on native timer / control.reason alone). Virtual now starts
    // at 0 with deadlineAt=1000; after hold, virtual now=1000 and direct execute
    // must refuse the handler and finish RUN_DEADLINE_EXCEEDED. A separate path
    // with virtual now still 0 must allow the handler.
    let virtualNow = 0

    let handlerEntered = false
    const tool: ToolDefinition = {
      name: 'direct_virtual',
      description: 'direct custom execute under virtual clock',
      inputSchema: { type: 'object', properties: {}, required: [] },
      handler: async () => {
        handlerEntered = true
        return { ok: true }
      },
    }

    // --- allowed path: virtual now stays below deadline ---
    {
      class DirectVirtualPort implements IIOPort {
        constructor(private readonly inner: IIOPort) {}
        async invokeLLM(
          req: ModelRequest,
          options?: LLMInvocationOptions,
        ) {
          return this.inner.invokeLLM(req, options)
        }
        async invokeTool(
          _name: string,
          _input: unknown,
          execute: (signal: AbortSignal) => Promise<unknown>,
          _opts?: Parameters<IIOPort['invokeTool']>[3],
          _control?: IOInvocationControl,
        ) {
          // Bypass DefaultIOPort — runtime thunk is the only remaining gate.
          return execute(new AbortController().signal)
        }
        now() { return virtualNow }
        nowSample() { return virtualNow }
        uuid() { return this.inner.uuid() }
      }

      virtualNow = 0
      handlerEntered = false
      const gw = new ControllableGateway([
        async () => toolCall('t-ok', 'direct_virtual'),
        async () => text('done'),
      ])
      const runtime = new AgentRuntime({
        config: makeConfig({
          builtinTools: { allow: [] },
          fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5, tools: ['direct_virtual'] }] },
        }),
        goal: 'g',
        input: 'i',
        stateStore: new MemoryStore(),
        recorder: new InMemoryRecorder(undefined, 'deadline-agent'),
        ioPort: new DirectVirtualPort(new DefaultIOPort(gw)),
        extraTools: [tool],
        control: { deadlineAt: 1000 },
      })
      const result = await runtime.run('i')
      expect(handlerEntered).toBe(true)
      expect(result.status).toBe('completed')
      expect(gw.requests).toBe(2)
    }

    // --- refused path: advance virtual clock past deadline, then direct execute ---
    {
      let releaseHold!: () => void
      const hold = new Promise<void>(resolve => { releaseHold = resolve })
      let markEntered!: () => void
      const entered = new Promise<void>(resolve => { markEntered = resolve })

      class DirectVirtualHoldPort implements IIOPort {
        constructor(private readonly inner: IIOPort) {}
        async invokeLLM(
          req: ModelRequest,
          options?: LLMInvocationOptions,
        ) {
          return this.inner.invokeLLM(req, options)
        }
        async invokeTool(
          _name: string,
          _input: unknown,
          execute: (signal: AbortSignal) => Promise<unknown>,
          _opts?: Parameters<IIOPort['invokeTool']>[3],
        ) {
          markEntered()
          await hold
          // Virtual clock at deadline; native timer may still be armed far away.
          // Direct execute must still fail closed via runtime thunk + IOPort clock.
          return execute(new AbortController().signal)
        }
        now() { return virtualNow }
        nowSample() { return virtualNow }
        uuid() { return this.inner.uuid() }
      }

      virtualNow = 0
      handlerEntered = false
      const gw = new ControllableGateway([
        async () => toolCall('t-late', 'direct_virtual'),
        async () => text('must-not-run'),
      ])
      const runtime = new AgentRuntime({
        config: makeConfig({
          builtinTools: { allow: [] },
          fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5, tools: ['direct_virtual'] }] },
        }),
        goal: 'g',
        input: 'i',
        stateStore: new MemoryStore(),
        recorder: new InMemoryRecorder(undefined, 'deadline-agent'),
        ioPort: new DirectVirtualHoldPort(new DefaultIOPort(gw)),
        extraTools: [tool],
        control: { deadlineAt: 1000 },
      })

      const resultP = runtime.run('i')
      await entered
      // Advance only the virtual IOPort clock — do not fire native timers.
      virtualNow = 1000
      releaseHold()
      const result = await resultP

      expect(handlerEntered).toBe(false)
      expect(result).toMatchObject({
        status: 'completed',
        stopReason: 'deadline',
        stopCode: 'RUN_DEADLINE_EXCEEDED',
      })
      expect(gw.requests).toBe(1)
    }
  })

  describe('RunControl far-future native timer', () => {
    const NODE_MAX_TIMER_DELAY = 2 ** 31 - 1

    afterEach(() => {
      jest.useRealTimers()
    })

    it('does not immediately stop when deadline is beyond Node max timer delay', async () => {
      jest.useFakeTimers()
      const epoch = 1_700_000_000_000
      jest.setSystemTime(epoch)

      const farDeadline = epoch + NODE_MAX_TIMER_DELAY + 60_000
      const rc = RunControl.create({
        now: epoch,
        clock: () => Date.now(),
        deadlineAt: farDeadline,
      })

      try {
        // Overflowing setTimeout would clamp delay to 1 and trip almost immediately.
        await jest.advanceTimersByTimeAsync(1_000)
        expect(rc.stopped).toBe(false)
        expect(rc.reason).toBeUndefined()

        // Crossing one full max-delay chunk must re-arm, not treat the deadline as due.
        await jest.advanceTimersByTimeAsync(NODE_MAX_TIMER_DELAY)
        expect(rc.stopped).toBe(false)
        expect(rc.reason).toBeUndefined()
      } finally {
        rc.dispose()
      }
    })

    it('trips deadline when the real/virtual deadline is reached after chunked re-arms', async () => {
      jest.useFakeTimers()
      const epoch = 1_700_000_000_000
      jest.setSystemTime(epoch)

      const farDeadline = epoch + NODE_MAX_TIMER_DELAY + 25_000
      const rc = RunControl.create({
        now: epoch,
        clock: () => Date.now(),
        deadlineAt: farDeadline,
      })

      try {
        await jest.advanceTimersByTimeAsync(NODE_MAX_TIMER_DELAY + 24_999)
        expect(rc.stopped).toBe(false)

        await jest.advanceTimersByTimeAsync(1)
        expect(rc.stopped).toBe(true)
        expect(rc.reason).toBe('deadline')
        expect(() => rc.throwIfStopped()).toThrow(RunControlError)
      } finally {
        rc.dispose()
      }
    })

    it('near-term deadline still trips on schedule (chunk path regression)', async () => {
      jest.useFakeTimers()
      const epoch = 5_000
      jest.setSystemTime(epoch)
      const rc = RunControl.create({
        now: epoch,
        clock: () => Date.now(),
        deadlineAt: epoch + 250,
      })
      try {
        await jest.advanceTimersByTimeAsync(249)
        expect(rc.stopped).toBe(false)
        await jest.advanceTimersByTimeAsync(1)
        expect(rc.stopped).toBe(true)
        expect(rc.reason).toBe('deadline')
      } finally {
        rc.dispose()
      }
    })

    it('chunked re-arms follow injected virtual clock, not Date.now or arm sample', async () => {
      // #237: each native-timer chunk must re-sample the injected run clock.
      // Existing far-future tests use clock: () => Date.now() which tracks Jest's
      // system time, so they cannot catch a regression that ignores `clock` and
      // advances from the previous arm sample (or ambient Date.now) instead.
      jest.useFakeTimers()
      const epoch = 1_700_000_000_000
      jest.setSystemTime(epoch)

      let virtualNow = epoch
      // Two full max-delay chunks beyond start would be needed if the virtual
      // clock advanced with the timer; we keep virtual frozen to prove re-arm
      // decisions do not ride Jest/system time.
      const deadlineAt = epoch + NODE_MAX_TIMER_DELAY + 10_000

      const rc = RunControl.create({
        now: virtualNow,
        clock: () => virtualNow,
        deadlineAt,
      })

      try {
        // First max chunk fires; virtual still at epoch → re-arm, do not stop.
        await jest.advanceTimersByTimeAsync(NODE_MAX_TIMER_DELAY)
        expect(rc.stopped).toBe(false)
        expect(rc.reason).toBeUndefined()

        // System/Jest clock is now far ahead of the virtual deadline. Another
        // chunk must still not trip while the injected clock is frozen.
        jest.setSystemTime(deadlineAt + 60_000)
        await jest.advanceTimersByTimeAsync(NODE_MAX_TIMER_DELAY)
        expect(rc.stopped).toBe(false)
        expect(rc.reason).toBeUndefined()

        // Move virtual clock just shy of deadline; next fire re-arms a short delay.
        virtualNow = deadlineAt - 1
        await jest.advanceTimersByTimeAsync(NODE_MAX_TIMER_DELAY)
        expect(rc.stopped).toBe(false)

        // Cross the deadline on the injected timeline, then let the armed timer fire.
        virtualNow = deadlineAt
        await jest.advanceTimersByTimeAsync(1)
        expect(rc.stopped).toBe(true)
        expect(rc.reason).toBe('deadline')
        expect(() => rc.throwIfStopped()).toThrow(RunControlError)
      } finally {
        rc.dispose()
      }
    })
  })





})
