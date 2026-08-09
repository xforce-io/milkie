import { v4 as uuidv4 } from 'uuid'
import {
  IOControlError,
  IOInvocationValidationError,
  type IModelGateway,
  type IOControlOperation,
  type IOInvocationControl,
  type ModelEvent,
  type ModelRequest,
  type ModelResponse,
} from '../types/model.js'
import type { InvalidToolArguments } from '../types/common.js'
import type { LineageBuffer } from '../trace/types.js'
import { aggregateStream } from '../gateway/StreamAggregator.js'

/**
 * IOPort — the Agent Runtime's declared boundary for non-deterministic
 * effects.
 *
 * All non-deterministic operations — LLM calls, tool invocations, clock reads,
 * UUID generation — pass through IOPort. This is the target architecture's
 * decoration point: future Agent Trace (event log + content-addressed cache +
 * non-determinism log) plugs in here to record events and serve cached
 * responses during replay, without the rest of the runtime knowing.
 *
 * Per ARCHITECTURE.md cross-cutting invariants:
 *   - "IOPort is part of Agent Runtime's design, not an Agent Trace-imposed
 *      hook." (#2)
 *   - "Agent Runtime does not depend on Agent Trace. Runtime depends only on
 *      its IOPort contract." (#3)
 *
 * Implementations:
 *   - DefaultIOPort: passthrough to gateway / Date.now() / uuid() — current
 *     production behavior.
 *   - (target) RecordingIOPort: writes llm.requested / llm.responded events
 *     plus non-determinism events to an Agent Trace event log.
 *   - (target) ReplayIOPort: serves cached responses by request hash, returns
 *     recorded clock / UUID values from the non-determinism log.
 */
const RESOLVED_IO_CONTROL = Symbol('ResolvedIOInvocationControl')
const NEVER_ABORTED_SIGNAL = new AbortController().signal
const MAX_TIMER_DELAY_MS = 2_147_483_647

export interface ResolvedIOInvocationControl extends IOInvocationControl {
  readonly [RESOLVED_IO_CONTROL]: true
}

export interface LLMInvocationOptions {
  readonly onEvent?: (event: ModelEvent) => void
  readonly control?: IOInvocationControl
}

export interface ToolInvocationOptions {
  readonly toolCallId?:       string
  readonly lineage?:          LineageBuffer
  readonly invalidArguments?: InvalidToolArguments
  readonly control?:          IOInvocationControl
}

/**
 * Validate and snapshot public invocation control before any asynchronous or
 * effectful work. Passing an already-resolved snapshot is intentionally
 * idempotent so decorators preserve one object across the whole call graph.
 */
export function resolveIOInvocationControl(
  control?: IOInvocationControl,
): ResolvedIOInvocationControl | undefined {
  if (!control) return undefined
  const resolved = control as ResolvedIOInvocationControl
  if (resolved[RESOLVED_IO_CONTROL]) return resolved
  if (control.deadlineAt !== undefined &&
      (!Number.isFinite(control.deadlineAt) || control.deadlineAt < 0)) {
    throw new IOInvocationValidationError()
  }
  return Object.freeze({
    ...(control.signal ? { signal: control.signal } : {}),
    ...(control.deadlineAt !== undefined ? { deadlineAt: control.deadlineAt } : {}),
    [RESOLVED_IO_CONTROL]: true as const,
  })
}

export function assertIOInvocationControl(
  control: ResolvedIOInvocationControl | undefined,
  operation: IOControlOperation,
  now = Date.now(),
): void {
  if (control?.signal?.aborted) {
    throw new IOControlError('IO_CANCELLED', operation)
  }
  if (control?.deadlineAt !== undefined && now >= control.deadlineAt) {
    throw new IOControlError('IO_DEADLINE_EXCEEDED', operation)
  }
}

function scheduleDeadline(deadlineAt: number, onDeadline: () => void): () => void {
  let timer: ReturnType<typeof setTimeout> | undefined
  let cancelled = false
  const schedule = () => {
    if (cancelled) return
    const remaining = deadlineAt - Date.now()
    if (remaining <= 0) {
      onDeadline()
      return
    }
    timer = setTimeout(schedule, Math.min(remaining, MAX_TIMER_DELAY_MS))
  }
  schedule()
  return () => {
    cancelled = true
    clearTimeout(timer)
  }
}

type ControlledExecution<T> = (
  signal: AbortSignal,
  isLatched: () => boolean,
) => Promise<T>

function invokeControlled<T>(
  operation: IOControlOperation,
  control: ResolvedIOInvocationControl | undefined,
  execute: ControlledExecution<T>,
): Promise<T> {
  assertIOInvocationControl(control, operation)
  if (!control?.signal && control?.deadlineAt === undefined) {
    return execute(NEVER_ABORTED_SIGNAL, () => false)
  }

  const effectiveController = new AbortController()
  let latched = false
  let cancelDeadline: (() => void) | undefined
  let onCallerAbort: (() => void) | undefined

  return new Promise<T>((resolve, reject) => {
    const cleanup = () => {
      cancelDeadline?.()
      if (onCallerAbort && control.signal) {
        control.signal.removeEventListener('abort', onCallerAbort)
      }
    }
    const settleControl = (code: 'IO_CANCELLED' | 'IO_DEADLINE_EXCEEDED') => {
      if (latched) return
      latched = true
      cleanup()
      effectiveController.abort()
      reject(new IOControlError(code, operation))
    }
    const settleSuccess = (value: T) => {
      if (latched) return
      latched = true
      cleanup()
      resolve(value)
    }
    const settleFailure = (error: unknown) => {
      if (latched) return
      latched = true
      cleanup()
      reject(error)
    }

    if (control.signal) {
      onCallerAbort = () => settleControl('IO_CANCELLED')
      control.signal.addEventListener('abort', onCallerAbort, { once: true })
    }
    if (control.deadlineAt !== undefined) {
      cancelDeadline = scheduleDeadline(
        control.deadlineAt,
        () => settleControl('IO_DEADLINE_EXCEEDED'),
      )
    }
    if (latched) return

    try {
      execute(effectiveController.signal, () => latched).then(settleSuccess, settleFailure)
    } catch (error) {
      settleFailure(error)
    }
  })
}

export function waitWithIOInvocationControl(
  delayMs: number,
  control: ResolvedIOInvocationControl | undefined,
  operation: IOControlOperation,
): Promise<void> {
  assertIOInvocationControl(control, operation)
  if (!control?.signal && control?.deadlineAt === undefined) {
    return new Promise(resolve => setTimeout(resolve, delayMs))
  }

  return new Promise<void>((resolve, reject) => {
    let settled = false
    let delayTimer: ReturnType<typeof setTimeout> | undefined
    let cancelDeadline: (() => void) | undefined
    const cleanup = () => {
      clearTimeout(delayTimer)
      cancelDeadline?.()
      control.signal?.removeEventListener('abort', onCallerAbort)
    }
    const settle = (error?: IOControlError) => {
      if (settled) return
      settled = true
      cleanup()
      if (error) reject(error)
      else resolve()
    }
    const onCallerAbort = () => settle(new IOControlError('IO_CANCELLED', operation))

    control.signal?.addEventListener('abort', onCallerAbort, { once: true })
    if (control.deadlineAt !== undefined) {
      cancelDeadline = scheduleDeadline(
        control.deadlineAt,
        () => settle(new IOControlError('IO_DEADLINE_EXCEEDED', operation)),
      )
    }
    if (settled) return
    delayTimer = setTimeout(() => settle(), delayMs)
  })
}

export interface IIOPort {
  invokeLLM(
    request: ModelRequest,
    options?: LLMInvocationOptions,
  ): Promise<ModelResponse>

  invokeTool(
    toolName: string,
    input: unknown,
    execute: (signal: AbortSignal) => Promise<unknown>,
    options?: ToolInvocationOptions,
  ): Promise<unknown>

  now(): number
  uuid(): string
}

export class DefaultIOPort implements IIOPort {
  constructor(private readonly gateway: IModelGateway) {}

  async invokeLLM(
    request: ModelRequest,
    options?: LLMInvocationOptions,
  ): Promise<ModelResponse> {
    const control = resolveIOInvocationControl(options?.control)
    return invokeControlled('llm', control, async (signal, isLatched) => {
      const gatewayOptions = signal === NEVER_ABORTED_SIGNAL ? undefined : { signal }
      if (options?.onEvent) {
        return aggregateStream(
          this.gateway.stream(request, gatewayOptions),
          options.onEvent,
          signal === NEVER_ABORTED_SIGNAL ? undefined : { signal, isLatched },
        )
      }
      return this.gateway.complete(request, gatewayOptions)
    })
  }

  async invokeTool(
    _toolName: string,
    _input: unknown,
    execute: (signal: AbortSignal) => Promise<unknown>,
    options?: ToolInvocationOptions,
  ): Promise<unknown> {
    const control = resolveIOInvocationControl(options?.control)
    return invokeControlled('tool', control, async signal => execute(signal))
  }

  now(): number {
    return Date.now()
  }

  uuid(): string {
    return uuidv4()
  }
}
