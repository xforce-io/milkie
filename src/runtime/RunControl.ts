import type {
  RunCancelledErrorEnvelope,
  RunDeadlineExceededErrorEnvelope,
} from '../types/model.js'

export type RunStopReason = 'deadline' | 'cancelled'

export interface RunControlCreateOptions {
  /** Epoch ms from IOPort.now() at run start. */
  now: number
  /**
   * Live run clock for segmented native-timer re-arms. Must share the IOPort
   * timeline (prefer nowSample so recording/replay nondet stays aligned).
   * When omitted, re-arms advance from the previous arm sample by the armed delay.
   */
  clock?: () => number
  deadlineAt?: number
  signal?: AbortSignal
  /** Parent run control — child inherits cancel and the earlier deadline. */
  parent?: RunControl
}

/** Node setTimeout delay is a signed 32-bit int; larger values overflow to 1ms. */
const MAX_NATIVE_TIMER_DELAY_MS = 2 ** 31 - 1

/**
 * #237: per-run cooperative cancellation boundary.
 *
 * Combines caller AbortSignal + absolute deadlineAt into one AbortSignal and a
 * single earliest-observed stop reason. Runtime checks this before scheduling
 * new model/tool I/O; in-flight work receives the same signal.
 */
export class RunControlError extends Error {
  readonly envelope: RunDeadlineExceededErrorEnvelope | RunCancelledErrorEnvelope
  readonly reason: RunStopReason

  constructor(reason: RunStopReason) {
    const envelope = reason === 'deadline'
      ? {
          code:      'RUN_DEADLINE_EXCEEDED' as const,
          message:   'Run deadline exceeded',
          phase:     'agent_loop' as const,
          retryable: true as const,
        }
      : {
          code:      'RUN_CANCELLED' as const,
          message:   'Run cancelled by caller',
          phase:     'agent_loop' as const,
          retryable: true as const,
        }
    super(envelope.message)
    this.name = 'RunControlError'
    this.reason = reason
    this.envelope = envelope
  }
}

export class RunControl {
  private readonly controller = new AbortController()
  private stopReason: RunStopReason | undefined
  private deadlineTimer: ReturnType<typeof setTimeout> | undefined
  private readonly externalCleanups: Array<() => void> = []
  private readonly clock: (() => number) | undefined
  readonly deadlineAt: number | undefined

  private constructor(deadlineAt: number | undefined, clock: (() => number) | undefined) {
    this.deadlineAt = deadlineAt
    this.clock = clock
  }

  static create(opts: RunControlCreateOptions): RunControl {
    let deadlineAt = opts.deadlineAt
    if (opts.parent?.deadlineAt !== undefined) {
      deadlineAt = deadlineAt === undefined
        ? opts.parent.deadlineAt
        : Math.min(deadlineAt, opts.parent.deadlineAt)
    }

    if (deadlineAt !== undefined) {
      if (typeof deadlineAt !== 'number' || !Number.isFinite(deadlineAt)) {
        throw new Error('control.deadlineAt must be a finite epoch-milliseconds number')
      }
    }

    const rc = new RunControl(deadlineAt, opts.clock)

    const arm = (reason: RunStopReason) => {
      rc.trip(reason)
    }

    // Parent stop is inherited with the parent's adjudicated reason when known.
    if (opts.parent) {
      if (opts.parent.stopped) {
        arm(opts.parent.reason ?? 'cancelled')
      } else {
        const onParent = () => arm(opts.parent!.reason ?? 'cancelled')
        opts.parent.signal.addEventListener('abort', onParent, { once: true })
        rc.externalCleanups.push(() => opts.parent!.signal.removeEventListener('abort', onParent))
      }
    }

    if (opts.signal) {
      if (opts.signal.aborted) {
        arm('cancelled')
      } else {
        const onExternal = () => arm('cancelled')
        opts.signal.addEventListener('abort', onExternal, { once: true })
        rc.externalCleanups.push(() => opts.signal!.removeEventListener('abort', onExternal))
      }
    }

    if (deadlineAt !== undefined && !rc.stopped) {
      rc.armDeadlineTimer(opts.now)
    }

    return rc
  }

  get signal(): AbortSignal {
    return this.controller.signal
  }

  get stopped(): boolean {
    return this.stopReason !== undefined
  }

  get reason(): RunStopReason | undefined {
    return this.stopReason
  }

  /** IOPort control snapshot for a single invocation. */
  invocationControl(): {
    signal: AbortSignal
    deadlineAt?: number
    reason?: RunStopReason
  } {
    return {
      signal: this.controller.signal,
      ...(this.deadlineAt !== undefined ? { deadlineAt: this.deadlineAt } : {}),
      ...(this.stopReason !== undefined ? { reason: this.stopReason } : {}),
    }
  }

  /**
   * Earliest-reason gate used at FSM / LLM / tool scheduling boundaries.
   * Throws RunControlError when the run must not start new external work.
   */
  throwIfStopped(): void {
    if (!this.stopReason) return
    throw new RunControlError(this.stopReason)
  }

  /**
   * Last-chance gate at handler start. `now` is the run's IOPort clock
   * (nowSample/now) so a virtual clock that has crossed deadlineAt trips
   * this run as deadline even when the native timer has not fired.
   */
  assertInvocationAllowed(now?: number): void {
    if (this.stopReason) throw new RunControlError(this.stopReason)
    if (
      this.deadlineAt !== undefined
      && now !== undefined
      && Number.isFinite(this.deadlineAt)
      && now >= this.deadlineAt
    ) {
      this.trip('deadline')
      throw new RunControlError('deadline')
    }
  }

  /**
   * Map an abort-shaped failure from in-flight I/O onto the adjudicated run stop.
   * Only rewrites when this RunControl has already stopped; idle control leaves
   * the original error intact so provider/tool AbortError is not misattributed.
   */
  mergeAbort(err: unknown): never {
    if (this.stopReason) throw new RunControlError(this.stopReason)
    throw err
  }

  dispose(): void {
    if (this.deadlineTimer !== undefined) {
      clearTimeout(this.deadlineTimer)
      this.deadlineTimer = undefined
    }
    for (const cleanup of this.externalCleanups.splice(0)) cleanup()
  }

  /**
   * Arm (or re-arm) the native deadline timer in chunks of at most
   * MAX_NATIVE_TIMER_DELAY_MS so far-future absolute deadlines never overflow
   * to a 1ms timeout. Each fire re-samples the run clock and either trips or
   * schedules the next chunk until deadlineAt.
   */
  private armDeadlineTimer(now: number): void {
    if (this.deadlineAt === undefined || this.stopReason !== undefined) return

    if (this.deadlineTimer !== undefined) {
      clearTimeout(this.deadlineTimer)
      this.deadlineTimer = undefined
    }

    const remaining = this.deadlineAt - now
    if (!(remaining > 0)) {
      this.trip('deadline')
      return
    }

    const delay = remaining > MAX_NATIVE_TIMER_DELAY_MS
      ? MAX_NATIVE_TIMER_DELAY_MS
      : remaining

    this.deadlineTimer = setTimeout(() => {
      this.deadlineTimer = undefined
      // Prefer the live run clock; without one, advance from the arm sample by
      // the delay we actually scheduled (correct when the timer subsystem and
      // clock move together, e.g. jest fake timers + Date.now).
      const sample = this.clock ? this.clock() : now + delay
      this.armDeadlineTimer(sample)
    }, delay)

    // Don't keep the process alive solely for a run deadline timer.
    if (typeof this.deadlineTimer === 'object' && this.deadlineTimer !== null && 'unref' in this.deadlineTimer) {
      ;(this.deadlineTimer as NodeJS.Timeout).unref?.()
    }
  }

  private trip(reason: RunStopReason): void {
    if (this.stopReason) return
    this.stopReason = reason
    if (this.deadlineTimer !== undefined) {
      clearTimeout(this.deadlineTimer)
      this.deadlineTimer = undefined
    }
    if (!this.controller.signal.aborted) {
      this.controller.abort()
    }
  }
}

export function isAbortError(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false
  const e = err as { name?: string; code?: string; message?: string }
  if (e.name === 'AbortError' || e.name === 'APIUserAbortError') return true
  if (e.code === 'ABORT_ERR') return true
  if (typeof e.message === 'string' && /aborted|abort/i.test(e.message)) return true
  return false
}

export function runControlErrorEnvelope(
  err: unknown,
): RunDeadlineExceededErrorEnvelope | RunCancelledErrorEnvelope | undefined {
  return err instanceof RunControlError ? err.envelope : undefined
}
