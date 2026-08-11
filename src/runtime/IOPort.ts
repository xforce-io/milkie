import { v4 as uuidv4 } from 'uuid'
import type { ModelRequest, ModelResponse, ModelEvent, IModelGateway } from '../types/model.js'
import type { InvalidToolArguments } from '../types/common.js'
import type { LineageBuffer } from '../trace/types.js'
import { aggregateStream } from '../gateway/StreamAggregator.js'
import { assertImageRequestSupported } from '../gateway/imageContent.js'
import { RunControlError } from './RunControl.js'


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
export interface ToolInvocationOptions {
  toolCallId?:       string
  lineage?:          LineageBuffer
  invalidArguments?: InvalidToolArguments
}

/** #237: optional cancel/deadline control for one LLM or tool invocation. */
export interface IOInvocationControl {
  signal?: AbortSignal
  deadlineAt?: number
  /**
   * Adjudicated stop reason when RunControl has already tripped.
   * Lets final gates rethrow the correct terminal without a fresh clock sample
   * (important for Runtime execute thunks under RecordingIOPort).
   */
  reason?: 'deadline' | 'cancelled'
}

export interface IIOPort {
  /**
   * Invoke a language model.
   * Implementations that support cancellation pass `control.signal` through to
   * the underlying gateway/SDK when available.
   */
  invokeLLM(
    request: ModelRequest,
    onEvent?: (e: ModelEvent) => void,
    control?: IOInvocationControl,
  ): Promise<ModelResponse>

  /**
   * Invoke a tool. The `execute` thunk is what actually runs the tool's
   * handler (or any other side-effecting work). Recording/replay ports wrap
   * this to log request/response or serve a cached result without re-running
   * the handler.
   *
   * `opts` carries optional pairing / lineage metadata. `control` carries the
   * run-level cancel signal for cooperative tool handlers.
   */
  invokeTool(
    toolName: string,
    input: unknown,
    execute: () => Promise<unknown>,
    opts?: ToolInvocationOptions,
    control?: IOInvocationControl,
  ): Promise<unknown>

  /** Current epoch milliseconds. Replacement for direct `Date.now()`. */
  now(): number

  /**
   * Infrastructure clock sample sharing now()'s timeline without agent-observable
   * nondet side effects. Recording ports must not emit clock.read; replay ports
   * must not consume the recorded clock queue. Used by run-control final gates
   * and segmented deadline timers. Optional for backward-compatible custom ports
   * — callers may fall back to now() when absent.
   */
  nowSample?(): number

  /** A new UUID. Replacement for direct `uuid()`. */
  uuid(): string
}

/**
 * Default IOPort: direct passthrough. No recording, no caching, no replay.
 * This is what production runs use today; Agent Trace decoration is added
 * by wrapping or replacing this implementation.
 */
export class DefaultIOPort implements IIOPort {
  constructor(private readonly gateway: IModelGateway) {}

  async invokeLLM(
    request: ModelRequest,
    onEvent?: (e: ModelEvent) => void,
    control?: IOInvocationControl,
  ): Promise<ModelResponse> {
    // #236: fail-closed for custom / undeclared gateways. Adapters also guard,
    // but every live LLM call must pass this I/O boundary before complete/stream.
    assertImageRequestSupported(request, this.gateway.capabilities, {
      provider: 'gateway',
      model: request.model,
    })
    const gwOpts = control?.signal ? { signal: control.signal } : undefined
    if (onEvent) {
      return aggregateStream(this.gateway.stream(request, gwOpts), onEvent)
    }
    return this.gateway.complete(request, gwOpts)
  }

  async invokeTool(
    _toolName: string,
    _input: unknown,
    execute: () => Promise<unknown>,
    _opts?: ToolInvocationOptions,
    control?: IOInvocationControl,
  ): Promise<unknown> {
    // #237: fail closed at the actual tool-start boundary. Runtime may have
    // passed its pre-gate earlier; wrappers (Recording/custom) can await before
    // dispatching. Re-check control immediately before the handler thunk runs.
    // Deadline comparison MUST use this port's clock, not ambient Date.now().
    assertToolInvocationAllowed(control, this.nowSample())
    return execute()
  }


  now(): number {
    return Date.now()
  }

  nowSample(): number {
    return this.now()
  }

  uuid(): string {
    return uuidv4()
  }
}

/**
 * #237: last-chance gate immediately before a tool handler thunk starts.
 * Uses the invocation control snapshot (signal + absolute deadlineAt + optional
 * adjudicated reason) and an optional IOPort clock sample for deadline
 * comparison — never ambient Date.now() — so virtual/custom clocks stay
 * consistent with RunControl.
 *
 * Precedence:
 *   1. control.reason when RunControl already tripped
 *   2. deadlineAt vs `now` when a clock sample is provided
 *   3. signal.aborted → cancelled
 *
 * Ports that own a clock (DefaultIOPort / RecordingIOPort) MUST pass `now` via
 * nowSample()/inner clock. Runtime execute thunks MUST also pass a sample from
 * the run's IOPort clock (prefer nowSample so Recording does not log clock.read;
 * replay never re-enters execute). Omitting `now` only checks reason/signal and
 * is insufficient against virtual-clock deadline bypass.
 *
 * @param control run-level cancel/deadline snapshot (optional)
 * @param now     epoch ms from IIOPort.nowSample()/now() at the gate instant (optional)
 */
export function assertToolInvocationAllowed(
  control?: IOInvocationControl,
  now?: number,
): void {
  if (!control) return
  if (control.reason === 'deadline' || control.reason === 'cancelled') {
    throw new RunControlError(control.reason)
  }
  const pastDeadline = control.deadlineAt !== undefined
    && Number.isFinite(control.deadlineAt)
    && now !== undefined
    && now >= control.deadlineAt
  if (pastDeadline) {
    throw new RunControlError('deadline')
  }
  if (control.signal?.aborted) {
    throw new RunControlError('cancelled')
  }
}
