import { DefaultIOPort, resolveIOInvocationControl } from '../runtime/IOPort'
import { RecordingIOPort } from '../trace/RecordingIOPort'
import { ReplayingIOPort } from '../trace/ReplayingIOPort'
import { CacheIndex } from '../trace/CacheIndex'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { hashModelRequest, hashToolCall } from '../trace/hash'
import {
  IOControlError,
  IOInvocationValidationError,
  type GatewayInvocationOptions,
  type IModelGateway,
  type ModelEvent,
  type ModelRequest,
  type ModelResponse,
} from '../types/model'
import type { Event, ToolRespondedPayload } from '../trace/types'

const REQUEST: ModelRequest = { model: 'test-model', messages: [] }
const RESPONSE: ModelResponse = {
  content: [{ type: 'text', text: 'done' }],
  toolCalls: [],
  finishReason: 'end_turn',
}

function deferred<T>(): {
  promise: Promise<T>
  resolve: (value: T) => void
  reject: (reason: unknown) => void
} {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

class DeferredGateway implements IModelGateway {
  readonly response = deferred<ModelResponse>()
  calls = 0
  signal?: AbortSignal

  async complete(_request: ModelRequest, options?: GatewayInvocationOptions): Promise<ModelResponse> {
    this.calls++
    this.signal = options?.signal
    return this.response.promise
  }

  async *stream(_request: ModelRequest, _options?: GatewayInvocationOptions): AsyncIterable<ModelEvent> {
    throw new Error('stream not expected')
  }
}

class NeverGateway implements IModelGateway {
  calls = 0

  async complete(): Promise<ModelResponse> {
    this.calls++
    return RESPONSE
  }

  async *stream(): AsyncIterable<ModelEvent> {
    this.calls++
    yield* []
  }
}

describe('IO invocation control', () => {
  afterEach(() => {
    jest.useRealTimers()
    jest.restoreAllMocks()
  })

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, -1])(
    'rejects invalid deadlineAt=%p with the stable validation error',
    (deadlineAt) => {
      expect(() => resolveIOInvocationControl({ deadlineAt })).toThrow(IOInvocationValidationError)
      expect(() => resolveIOInvocationControl({ deadlineAt })).toThrow(
        'I/O invocation deadline must be a finite non-negative Unix epoch millisecond.',
      )
    },
  )

  it('snapshots deadlineAt and keeps the caller signal reference', () => {
    const controller = new AbortController()
    const mutable = { signal: controller.signal, deadlineAt: 123 }
    const resolved = resolveIOInvocationControl(mutable)!

    mutable.deadlineAt = 456

    expect(resolved.deadlineAt).toBe(123)
    expect(resolved.signal).toBe(controller.signal)
    expect(resolveIOInvocationControl(resolved)).toBe(resolved)
  })

  it('treats deadlineAt=0 and now===deadlineAt as valid expired deadlines', async () => {
    const gateway = new NeverGateway()
    const port = new DefaultIOPort(gateway)

    await expect(port.invokeLLM(REQUEST, { control: { deadlineAt: 0 } })).rejects.toMatchObject({
      envelope: { code: 'IO_DEADLINE_EXCEEDED', operation: 'llm' },
    })

    jest.spyOn(Date, 'now').mockReturnValue(42)
    await expect(port.invokeTool('tool', {}, async () => 'unused', {
      control: { deadlineAt: 42 },
    })).rejects.toMatchObject({
      envelope: { code: 'IO_DEADLINE_EXCEEDED', operation: 'tool' },
    })
    expect(gateway.calls).toBe(0)
  })

  it('pre-cancel wins over an already expired deadline before execution', async () => {
    const controller = new AbortController()
    controller.abort()
    const execute = jest.fn(async () => 'unused')
    const port = new DefaultIOPort(new NeverGateway())

    await expect(port.invokeTool('tool', {}, execute, {
      control: { signal: controller.signal, deadlineAt: 0 },
    })).rejects.toMatchObject({
      envelope: {
        code: 'IO_CANCELLED',
        message: 'I/O invocation was cancelled.',
        phase: 'io_control',
        operation: 'tool',
        retryable: false,
      },
    })
    expect(execute).not.toHaveBeenCalled()
  })

  it('actively cancels an in-flight LLM, aborts the gateway signal, and ignores a late result', async () => {
    const gateway = new DeferredGateway()
    const port = new DefaultIOPort(gateway)
    const controller = new AbortController()
    const invocation = port.invokeLLM(REQUEST, { control: { signal: controller.signal } })

    expect(gateway.calls).toBe(1)
    expect(gateway.signal?.aborted).toBe(false)
    controller.abort('private caller reason')

    await expect(invocation).rejects.toBeInstanceOf(IOControlError)
    await expect(invocation).rejects.toMatchObject({
      envelope: { code: 'IO_CANCELLED', operation: 'llm', retryable: false },
    })
    expect(gateway.signal?.aborted).toBe(true)

    gateway.response.resolve(RESPONSE)
    await Promise.resolve()
  })

  it('settles a non-cooperative Tool at its absolute deadline and aborts its signal', async () => {
    jest.useFakeTimers()
    jest.setSystemTime(1_000)
    let signal: AbortSignal | undefined
    let settled = false
    const entered = deferred<void>()
    const deadlineAt = Date.now() + 20
    const invocation = new DefaultIOPort(new NeverGateway()).invokeTool(
      'slow-tool',
      {},
      async (effectiveSignal) => {
        signal = effectiveSignal
        entered.resolve()
        return new Promise<never>(() => {})
      },
      { control: { deadlineAt } },
    )
    void invocation.then(
      () => { settled = true },
      () => { settled = true },
    )

    await entered.promise
    expect(signal?.aborted).toBe(false)

    await jest.advanceTimersByTimeAsync(19)
    expect(settled).toBe(false)
    expect(signal?.aborted).toBe(false)

    await jest.advanceTimersByTimeAsync(1)
    await expect(invocation).rejects.toMatchObject({
      envelope: {
        code: 'IO_DEADLINE_EXCEEDED',
        message: 'I/O invocation deadline exceeded.',
        phase: 'io_control',
        operation: 'tool',
        retryable: false,
      },
    })
    expect(settled).toBe(true)
    expect(signal?.aborted).toBe(true)
    expect(Date.now()).toBe(deadlineAt)
  })

  it('cleans up a controlled stream exactly once and drops an event pulled after cancellation', async () => {
    const next = deferred<IteratorResult<ModelEvent>>()
    let returnCalls = 0
    const stream: AsyncIterable<ModelEvent> = {
      [Symbol.asyncIterator]() {
        return {
          next: () => next.promise,
          return: async () => {
            returnCalls++
            return { done: true, value: undefined }
          },
        }
      },
    }
    const gateway: IModelGateway = {
      async complete(): Promise<ModelResponse> { return RESPONSE },
      stream: (_request, _options) => stream,
    }
    const events: ModelEvent[] = []
    const controller = new AbortController()
    const invocation = new DefaultIOPort(gateway).invokeLLM(REQUEST, {
      onEvent: event => events.push(event),
      control: { signal: controller.signal },
    })

    controller.abort()
    await expect(invocation).rejects.toMatchObject({ envelope: { code: 'IO_CANCELLED' } })
    next.resolve({ done: false, value: { type: 'message_delta', data: { text: 'late' } } })
    await Promise.resolve()
    await Promise.resolve()

    expect(events).toEqual([])
    expect(returnCalls).toBe(1)
  })

  it('removes the caller abort listener after normal completion', async () => {
    const controller = new AbortController()
    const add = jest.spyOn(controller.signal, 'addEventListener')
    const remove = jest.spyOn(controller.signal, 'removeEventListener')
    const gateway: IModelGateway = {
      async complete(): Promise<ModelResponse> { return RESPONSE },
      async *stream(): AsyncIterable<ModelEvent> { yield* [] },
    }

    await new DefaultIOPort(gateway).invokeLLM(REQUEST, { control: { signal: controller.signal } })

    expect(add).toHaveBeenCalledWith('abort', expect.any(Function), expect.objectContaining({ once: true }))
    expect(remove).toHaveBeenCalledWith('abort', expect.any(Function))
  })
})

describe('IOPort decorator preflight', () => {
  it('rejects an invalid RecordingIOPort control before Trace, provider, or executor side effects', async () => {
    const store = new MemoryEventStore()
    const gateway = new NeverGateway()
    const recording = new RecordingIOPort(new DefaultIOPort(gateway), store, 'run-1')
    const execute = jest.fn(async () => 'unused')

    await expect(recording.invokeLLM(REQUEST, {
      control: { deadlineAt: Number.NaN },
    })).rejects.toBeInstanceOf(IOInvocationValidationError)
    await expect(recording.invokeTool('tool', {}, execute, {
      control: { deadlineAt: -1 },
    })).rejects.toBeInstanceOf(IOInvocationValidationError)

    expect(await store.readByRunId('run-1')).toEqual([])
    expect(gateway.calls).toBe(0)
    expect(execute).not.toHaveBeenCalled()
  })

  it('does not consume Replay FIFO for pre-cancelled or expired invocations', async () => {
    const toolInput = { value: 1 }
    const toolOutput = { ok: true }
    const llmHash = hashModelRequest(REQUEST)
    const toolHash = hashToolCall('tool', toolInput)
    const cachedLLM: Event = {
      id: 'llm-1',
      runId: 'run-1',
      type: 'llm.responded',
      actor: 'runtime',
      timestamp: 1,
      payload: { response: RESPONSE, requestHash: llmHash },
    }
    const cachedTool: Event<ToolRespondedPayload> = {
      id: 'tool-1',
      runId: 'run-1',
      type: 'tool.responded',
      actor: 'runtime',
      timestamp: 2,
      payload: { toolName: 'tool', output: toolOutput, requestHash: toolHash },
    }
    const cache = CacheIndex.fromEvents([cachedLLM, cachedTool])
    const replay = new ReplayingIOPort(cache, new DefaultIOPort(new NeverGateway()))
    const execute = jest.fn(async () => 'unused')
    const controller = new AbortController()
    controller.abort()

    await expect(replay.invokeLLM(REQUEST, {
      control: { signal: controller.signal },
    })).rejects.toMatchObject({ envelope: { code: 'IO_CANCELLED' } })
    await expect(replay.invokeTool('tool', toolInput, execute, {
      control: { signal: controller.signal },
    })).rejects.toMatchObject({ envelope: { code: 'IO_CANCELLED' } })
    expect(cache.remaining()).toMatchObject({ llm: 1, tool: 1 })
    expect(execute).not.toHaveBeenCalled()

    await expect(replay.invokeLLM(REQUEST, {
      control: { deadlineAt: 0 },
    })).rejects.toMatchObject({ envelope: { code: 'IO_DEADLINE_EXCEEDED' } })
    await expect(replay.invokeTool('tool', toolInput, execute, {
      control: { deadlineAt: 0 },
    })).rejects.toMatchObject({ envelope: { code: 'IO_DEADLINE_EXCEEDED' } })
    expect(cache.remaining()).toMatchObject({ llm: 1, tool: 1 })
    expect(execute).not.toHaveBeenCalled()

    await expect(replay.invokeLLM(REQUEST)).resolves.toEqual(RESPONSE)
    await expect(replay.invokeTool('tool', toolInput, execute)).resolves.toEqual(toolOutput)
    expect(cache.remaining()).toMatchObject({ llm: 0, tool: 0 })
    expect(execute).not.toHaveBeenCalled()
  })
})
