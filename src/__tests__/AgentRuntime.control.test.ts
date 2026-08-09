import { AgentRuntime } from '../runtime/AgentRuntime'
import { DefaultIOPort, type IIOPort, type LLMInvocationOptions, type ToolInvocationOptions } from '../runtime/IOPort'
import { MemoryStore } from '../store/MemoryStore'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder'
import type { AgentConfig } from '../types/agent'
import type {
  GatewayInvocationOptions,
  IModelGateway,
  IOInvocationControl,
  ModelEvent,
  ModelRequest,
  ModelResponse,
} from '../types/model'
import type { ToolDefinition } from '../types/tool'

interface Deferred<T> {
  promise: Promise<T>
  resolve: (value: T) => void
}

function config(states: AgentConfig['fsm']['states'], subAgents?: AgentConfig['subAgents']): AgentConfig {
  return {
    agentId: 'parent',
    version: '1.0.0',
    systemPrompt: 'test',
    fsm: { states },
    model: { provider: 'test', model: 'test', adapter: 'test' },
    ...(subAgents ? { subAgents } : {}),
  }
}

function textResponse(text: string): ModelResponse {
  return { content: [{ type: 'text', text }], toolCalls: [], finishReason: 'end_turn' }
}

function toolResponse(id: string, name: string, input: unknown = {}): ModelResponse {
  return {
    content: [{ type: 'tool_use', id, name, input }],
    toolCalls: [{ id, name, input }],
    finishReason: 'tool_use',
  }
}

function deferred<T = void>(): Deferred<T> {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(res => { resolve = res })
  return { promise, resolve }
}

class ScriptedGateway implements IModelGateway {
  private index = 0
  readonly signals: AbortSignal[] = []
  readonly entered = deferred()

  constructor(private readonly responses: Array<ModelResponse | Promise<ModelResponse>>) {}

  async complete(_request: ModelRequest, options?: GatewayInvocationOptions): Promise<ModelResponse> {
    if (options?.signal) this.signals.push(options.signal)
    this.entered.resolve()
    const response = this.responses[this.index++]
    if (!response) throw new Error('No scripted response')
    return response
  }

  async *stream(_request: ModelRequest, _options?: GatewayInvocationOptions): AsyncIterable<ModelEvent> {
    throw new Error('stream not expected')
  }
}

function runtimeWith(
  gateway: IModelGateway,
  runtimeConfig: AgentConfig,
  tools: ToolDefinition[] = [],
  control?: IOInvocationControl,
): AgentRuntime {
  return new AgentRuntime({
    config: runtimeConfig,
    goal: 'goal',
    input: 'input',
    stateStore: new MemoryStore(),
    recorder: new InMemoryRecorder(),
    ioPort: new DefaultIOPort(gateway),
    extraTools: tools,
    control,
  })
}

describe('AgentRuntime I/O control propagation', () => {
  afterEach(() => jest.useRealTimers())

  it('returns the stable LLM cancellation envelope and aborts the gateway signal', async () => {
    const pending = new Promise<ModelResponse>(() => {})
    const gateway = new ScriptedGateway([pending])
    const controller = new AbortController()
    const run = runtimeWith(gateway, config([{ name: 'react', type: 'llm' }]), [], {
      signal: controller.signal,
    }).run('input')

    await gateway.entered.promise
    controller.abort()
    const result = await run

    expect(gateway.signals).toHaveLength(1)
    expect(gateway.signals[0]?.aborted).toBe(true)
    expect(result).toMatchObject({
      status: 'error',
      error: {
        code: 'IO_CANCELLED',
        phase: 'io_control',
        operation: 'llm',
        retryable: false,
      },
    })
  })

  it('does not downgrade ordinary Tool cancellation into a ToolResult', async () => {
    const entered = deferred()
    let toolSignal: AbortSignal | undefined
    const tool: ToolDefinition = {
      name: 'slow',
      description: 'slow',
      inputSchema: { type: 'object', properties: {} },
      handler: async (_input, ctx) => {
        toolSignal = ctx.signal
        entered.resolve()
        return new Promise<never>(() => {})
      },
    }
    const gateway = new ScriptedGateway([toolResponse('call-1', 'slow')])
    const controller = new AbortController()
    const run = runtimeWith(gateway, config([{ name: 'react', type: 'llm' }]), [tool], {
      signal: controller.signal,
    }).run('input')

    await entered.promise
    controller.abort()
    const result = await run

    expect(toolSignal?.aborted).toBe(true)
    expect(result).toMatchObject({
      status: 'error',
      error: { code: 'IO_CANCELLED', operation: 'tool' },
    })
  })

  it('routes action-state handlers through IOPort with the effective ToolContext signal', async () => {
    const entered = deferred()
    let toolSignal: AbortSignal | undefined
    const tool: ToolDefinition = {
      name: 'action',
      description: 'action',
      inputSchema: { type: 'object', properties: {} },
      handler: async (_input, ctx) => {
        toolSignal = ctx.signal
        entered.resolve()
        return new Promise<never>(() => {})
      },
    }
    const controller = new AbortController()
    const run = runtimeWith(
      new ScriptedGateway([]),
      config([{ name: 'act', type: 'action', handler: 'action' }]),
      [tool],
      { signal: controller.signal },
    ).run('input')

    await entered.promise
    controller.abort()
    const result = await run

    expect(toolSignal?.aborted).toBe(true)
    expect(result).toMatchObject({
      status: 'error',
      error: { code: 'IO_CANCELLED', operation: 'tool' },
    })
  })

  it('cancels every in-flight parallel Tool and preserves the control failure', async () => {
    const enteredA = deferred()
    const enteredB = deferred()
    const signals: AbortSignal[] = []
    const makeTool = (name: string, entered: Deferred<void>): ToolDefinition => ({
      name,
      description: name,
      parallelSafe: true,
      inputSchema: { type: 'object', properties: {} },
      handler: async (_input, ctx) => {
        signals.push(ctx.signal)
        entered.resolve()
        return new Promise<never>(() => {})
      },
    })
    const gateway = new ScriptedGateway([{
      content: [
        { type: 'tool_use', id: 'a', name: 'a', input: {} },
        { type: 'tool_use', id: 'b', name: 'b', input: {} },
      ],
      toolCalls: [
        { id: 'a', name: 'a', input: {} },
        { id: 'b', name: 'b', input: {} },
      ],
      finishReason: 'tool_use',
    }])
    const controller = new AbortController()
    const run = runtimeWith(
      gateway,
      config([{ name: 'react', type: 'llm' }]),
      [makeTool('a', enteredA), makeTool('b', enteredB)],
      { signal: controller.signal },
    ).run('input')

    await Promise.all([enteredA.promise, enteredB.promise])
    controller.abort()
    const result = await run

    expect(signals).toHaveLength(2)
    expect(signals.every(signal => signal.aborted)).toBe(true)
    expect(result).toMatchObject({ status: 'error', error: { code: 'IO_CANCELLED', operation: 'tool' } })
  })

  it('cancels retry backoff without starting the next Tool attempt', async () => {
    const firstAttempt = deferred()
    jest.useFakeTimers()
    const releaseFailure = deferred()
    let attempts = 0
    const tool: ToolDefinition = {
      name: 'retryable',
      description: 'retryable',
      inputSchema: { type: 'object', properties: {} },
      handler: async () => {
        attempts++
        firstAttempt.resolve()
        await releaseFailure.promise
        throw Object.assign(new Error('retry me'), { retryable: true })
      },
    }
    const controller = new AbortController()
    const run = runtimeWith(
      new ScriptedGateway([toolResponse('retry-1', 'retryable')]),
      config([{ name: 'react', type: 'llm' }]),
      [tool],
      { signal: controller.signal },
    ).run('input')

    await firstAttempt.promise
    releaseFailure.resolve()
    await Promise.resolve()
    await jest.advanceTimersByTimeAsync(0)
    controller.abort()
    const result = await run

    expect(attempts).toBe(1)
    expect(result).toMatchObject({ status: 'error', error: { code: 'IO_CANCELLED', operation: 'tool' } })
  })

  it('passes one resolved control snapshot through parent LLM, Tool, and child LLM calls', async () => {
    const seenControls: Array<IOInvocationControl | undefined> = []
    const effectiveSignal = new AbortController().signal
    let nextUuid = 0
    const responses = [
      toolResponse('child-call', 'worker', { goal: 'child goal', input: 'child input' }),
      textResponse('child done'),
      textResponse('parent done'),
    ]
    const port: IIOPort = {
      async invokeLLM(_request: ModelRequest, options?: LLMInvocationOptions): Promise<ModelResponse> {
        seenControls.push(options?.control)
        const response = responses.shift()
        if (!response) throw new Error('No scripted response')
        return response
      },
      async invokeTool(
        _name: string,
        _input: unknown,
        execute: (signal: AbortSignal) => Promise<unknown>,
        options?: ToolInvocationOptions,
      ): Promise<unknown> {
        seenControls.push(options?.control)
        return execute(effectiveSignal)
      },
      now: () => Date.now(),
      uuid: () => `uuid-${++nextUuid}`,
    }
    const childConfig: AgentConfig = {
      ...config([{ name: 'react', type: 'llm' }]),
      agentId: 'worker',
    }
    const rawControl = { deadlineAt: Date.now() + 60_000 }
    const runtime = new AgentRuntime({
      config: config([{ name: 'react', type: 'llm' }], { worker: 'worker' }),
      goal: 'goal',
      input: 'input',
      stateStore: new MemoryStore(),
      recorder: new InMemoryRecorder(),
      ioPort: port,
      subAgentConfigs: new Map([['worker', childConfig]]),
      control: rawControl,
    })

    const result = await runtime.run('input')

    expect(result.status).toBe('completed')
    expect(seenControls).toHaveLength(4)
    expect(seenControls[0]).not.toBe(rawControl)
    expect(seenControls.every(control => control === seenControls[0])).toBe(true)
  })
})
