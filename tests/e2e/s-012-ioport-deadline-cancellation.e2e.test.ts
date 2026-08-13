import { Milkie } from '../../src/runtime/Milkie'
import { MemoryStore } from '../../src/store/MemoryStore'
import { MemoryEventStore } from '../../src/trace/MemoryEventStore'
import { IOInvocationValidationError } from '../../src/types/model'
import type { AgentConfig } from '../../src/types/agent'
import type {
  GatewayInvocationOptions,
  IModelGateway,
  ModelEvent,
  ModelRequest,
  ModelResponse,
} from '../../src/types/model'
import type { ToolDefinition } from '../../src/types/tool'

const LLM_CONFIG: AgentConfig = {
  agentId: 'controlled-llm',
  version: '1.0.0',
  systemPrompt: 'test',
  fsm: { states: [{ name: 'react', type: 'llm' }] },
  model: { provider: 'test', model: 'test', adapter: 'test' },
}

function toolConfig(agentId: string, state: AgentConfig['fsm']['states'][number]): AgentConfig {
  return {
    ...LLM_CONFIG,
    agentId,
    fsm: { states: [state] },
  }
}

function deferred<T = void>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(res => { resolve = res })
  return { promise, resolve }
}

class ControlledGateway implements IModelGateway {
  calls = 0
  signal?: AbortSignal
  readonly entered = deferred()
  response: Promise<ModelResponse> = new Promise<ModelResponse>(() => {})

  async complete(_request: ModelRequest, options?: GatewayInvocationOptions): Promise<ModelResponse> {
    this.calls++
    this.signal = options?.signal
    this.entered.resolve()
    return this.response
  }

  async *stream(_request: ModelRequest, _options?: GatewayInvocationOptions): AsyncIterable<ModelEvent> {
    throw new Error('stream not expected')
  }
}

describe('S1/S2: Milkie IOPort deadline and cancellation', () => {
  it('S1 terminates an in-flight LLM by deadline within the controlled 100ms tolerance', async () => {
    const gateway = new ControlledGateway()
    const milkie = new Milkie({ stateStore: new MemoryStore(), gateway })
    milkie.registerAgent(LLM_CONFIG)
    const deadlineAt = Date.now() + 100

    const result = await milkie.invoke({
      agentId: LLM_CONFIG.agentId,
      goal: 'wait',
      input: 'wait',
      control: { deadlineAt },
    })

    expect(gateway.calls).toBe(1)
    expect(gateway.signal?.aborted).toBe(true)
    expect(Date.now()).toBeLessThanOrEqual(deadlineAt + 100)
    expect(result).toMatchObject({
      status: 'completed',
      stopReason: 'deadline',
      stopCode: 'IO_DEADLINE_EXCEEDED',
    })
  })


  it('S2 actively cancels an in-flight LLM with a distinct stable result', async () => {
    const gateway = new ControlledGateway()
    const milkie = new Milkie({ stateStore: new MemoryStore(), gateway })
    milkie.registerAgent(LLM_CONFIG)
    const controller = new AbortController()
    const invocation = milkie.invoke({
      agentId: LLM_CONFIG.agentId,
      goal: 'wait',
      input: 'wait',
      control: { signal: controller.signal },
    })

    await gateway.entered.promise
    controller.abort()
    const result = await invocation

    expect(gateway.signal?.aborted).toBe(true)
    expect(result).toMatchObject({
      status: 'interrupted',
      stopReason: 'cancelled',
      stopCode: 'IO_CANCELLED',
    })
    expect(result.stopCode).not.toBe('MODEL_TIMEOUT')
  })
  it('S1 terminates an in-flight ordinary Tool by deadline and preserves operation=tool', async () => {
    const entered = deferred()
    let signal: AbortSignal | undefined
    const gateway: IModelGateway = {
      async complete(): Promise<ModelResponse> {
        return {
          content: [{ type: 'tool_use', id: 'slow-1', name: 'slow', input: {} }],
          toolCalls: [{ id: 'slow-1', name: 'slow', input: {} }],
          finishReason: 'tool_use',
        }
      },
      async *stream(): AsyncIterable<ModelEvent> { yield* [] },
    }
    const slow: ToolDefinition = {
      name: 'slow',
      description: 'slow',
      inputSchema: { type: 'object', properties: {} },
      handler: async (_input, ctx) => {
        signal = ctx.signal
        entered.resolve()
        return new Promise<never>(() => {})
      },
    }
    const milkie = new Milkie({ stateStore: new MemoryStore(), gateway, tools: [slow] })
    milkie.registerAgent(toolConfig('controlled-tool', { name: 'react', type: 'llm' }))
    const deadlineAt = Date.now() + 200
    const invocation = milkie.invoke({
      agentId: 'controlled-tool',
      goal: 'wait',
      input: 'wait',
      control: { deadlineAt },
    })

    await entered.promise
    const result = await invocation

    expect(signal?.aborted).toBe(true)
    expect(Date.now()).toBeLessThanOrEqual(deadlineAt + 100)
    expect(result).toMatchObject({
      status: 'completed',
      stopReason: 'deadline',
      stopCode: 'IO_DEADLINE_EXCEEDED',
    })
  })

  it('S2 actively cancels an in-flight ordinary Tool without continuing the LLM loop', async () => {
    const entered = deferred()
    let signal: AbortSignal | undefined
    let gatewayCalls = 0
    const gateway: IModelGateway = {
      async complete(): Promise<ModelResponse> {
        gatewayCalls++
        return {
          content: [{ type: 'tool_use', id: 'slow-cancel-1', name: 'slow-cancel', input: {} }],
          toolCalls: [{ id: 'slow-cancel-1', name: 'slow-cancel', input: {} }],
          finishReason: 'tool_use',
        }
      },
      async *stream(): AsyncIterable<ModelEvent> { yield* [] },
    }
    const slow: ToolDefinition = {
      name: 'slow-cancel',
      description: 'slow-cancel',
      inputSchema: { type: 'object', properties: {} },
      handler: async (_input, ctx) => {
        signal = ctx.signal
        entered.resolve()
        return new Promise<never>(() => {})
      },
    }
    const milkie = new Milkie({ stateStore: new MemoryStore(), gateway, tools: [slow] })
    milkie.registerAgent(toolConfig('cancelled-tool', { name: 'react', type: 'llm' }))
    const controller = new AbortController()
    const invocation = milkie.invoke({
      agentId: 'cancelled-tool',
      goal: 'wait',
      input: 'wait',
      control: { signal: controller.signal },
    })

    await entered.promise
    expect(signal?.aborted).toBe(false)
    controller.abort()
    const result = await invocation

    expect(signal?.aborted).toBe(true)
    expect(result).toMatchObject({
      status: 'interrupted',
      stopReason: 'cancelled',
      stopCode: 'IO_CANCELLED',
    })
    expect(result.stopCode).not.toBe('MODEL_TIMEOUT')
    expect(gatewayCalls).toBe(1)
  })

  it('S2 actively cancels an action-state Tool and exposes its effective signal', async () => {
    const entered = deferred()
    let signal: AbortSignal | undefined
    const action: ToolDefinition = {
      name: 'action',
      description: 'action',
      inputSchema: { type: 'object', properties: {} },
      handler: async (_input, ctx) => {
        signal = ctx.signal
        entered.resolve()
        return new Promise<never>(() => {})
      },
    }
    const gateway: IModelGateway = {
      async complete(): Promise<ModelResponse> { throw new Error('LLM not expected') },
      async *stream(): AsyncIterable<ModelEvent> { yield* [] },
    }
    const milkie = new Milkie({ stateStore: new MemoryStore(), gateway, tools: [action] })
    milkie.registerAgent(toolConfig('controlled-action', { name: 'act', type: 'action', handler: 'action' }))
    const controller = new AbortController()
    const invocation = milkie.invoke({
      agentId: 'controlled-action',
      goal: 'wait',
      input: 'wait',
      control: { signal: controller.signal },
    })

    await entered.promise
    controller.abort()
    const result = await invocation

    expect(signal?.aborted).toBe(true)
    expect(result).toMatchObject({
      status: 'interrupted',
      stopReason: 'cancelled',
      stopCode: 'IO_CANCELLED',
    })
  })

  it('rejects invalid control before creating Trace or invoking a provider', async () => {
    const eventStore = new MemoryEventStore()
    const append = jest.spyOn(eventStore, 'append')
    const gateway = new ControlledGateway()
    const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore, gateway })
    milkie.registerAgent(LLM_CONFIG)

    await expect(milkie.invoke({
      agentId: LLM_CONFIG.agentId,
      goal: 'invalid',
      input: 'invalid',
      control: { deadlineAt: Number.NaN },
    })).rejects.toBeInstanceOf(IOInvocationValidationError)

    expect(gateway.calls).toBe(0)
    expect(append).not.toHaveBeenCalled()
  })
  it('validates resume control before checkpoint or run side effects', async () => {
    const stateStore = new MemoryStore()
    const get = jest.spyOn(stateStore, 'get')
    const milkie = new Milkie({ stateStore, gateway: new ControlledGateway() })
    milkie.registerAgent(LLM_CONFIG)

    await expect(milkie.resume(
      'missing-checkpoint',
      LLM_CONFIG.agentId,
      'invalid',
      'invalid',
      { control: { deadlineAt: Number.POSITIVE_INFINITY } },
    )).rejects.toBeInstanceOf(IOInvocationValidationError)

    expect(get).not.toHaveBeenCalled()
  })
})
