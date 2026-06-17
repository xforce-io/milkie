/**
 * RunLifecycle — the lower, framework-fixed run-lifecycle state machine (#175).
 *
 * This is the "kept" half of the old FSM split (see docs/design/175): it models
 * a run's EXECUTION lifecycle, not any business topology. It knows nothing about
 * authored business states; it only consumes the four outcome signals a loop body
 * produces (plus interrupt) and the resume re-entry.
 *
 *   running ──continue──▶ running
 *   running ──need_input─▶ paused      (suspended, resumable)
 *   running ──interrupt──▶ interrupted (suspended, resumable)
 *   running ──done──────▶ completed    (terminal)
 *   running ──error─────▶ failed       (terminal)
 *   paused | interrupted ──resume──▶ running
 */

export type RunLifecycleState =
  | 'running'
  | 'paused'
  | 'interrupted'
  | 'completed'
  | 'failed'

export type RunSignal =
  | 'continue'
  | 'need_input'
  | 'done'
  | 'error'
  | 'interrupt'
  | 'resume'

export interface RunLifecycleSnapshot {
  state: RunLifecycleState
}

const ALL_STATES: ReadonlySet<RunLifecycleState> = new Set([
  'running', 'paused', 'interrupted', 'completed', 'failed',
])
const TERMINAL: ReadonlySet<RunLifecycleState>  = new Set(['completed', 'failed'])
const SUSPENDED: ReadonlySet<RunLifecycleState> = new Set(['paused', 'interrupted'])

export class RunLifecycle {
  private _state: RunLifecycleState = 'running'

  get state(): RunLifecycleState {
    return this._state
  }

  isTerminal(): boolean {
    return TERMINAL.has(this._state)
  }

  /** Serialize for the checkpoint (the `{lifecycle}` field; see §4 / D7). */
  snapshot(): RunLifecycleSnapshot {
    return { state: this._state }
  }

  /** Reconstruct from a checkpoint so a later run can resume. */
  static restore(snap: RunLifecycleSnapshot): RunLifecycle {
    if (!ALL_STATES.has(snap.state)) {
      throw new Error(`RunLifecycle.restore: unknown state "${snap.state}"`)
    }
    const lc = new RunLifecycle()
    lc._state = snap.state
    return lc
  }

  /** Apply an outcome signal, advancing the lifecycle. Returns the new state. */
  signal(sig: RunSignal): RunLifecycleState {
    if (this.isTerminal()) {
      throw new Error(`RunLifecycle: run is terminal (${this._state}); cannot apply signal "${sig}"`)
    }
    this._state = this.next(sig)
    return this._state
  }

  private next(sig: RunSignal): RunLifecycleState {
    const from = this._state

    if (sig === 'resume') {
      if (!SUSPENDED.has(from)) {
        throw new Error(`RunLifecycle: cannot resume from "${from}" — only paused/interrupted are suspended`)
      }
      return 'running'
    }

    // continue / need_input / done / error / interrupt are only valid while running.
    if (from !== 'running') {
      throw new Error(`RunLifecycle: signal "${sig}" is only valid in running, not "${from}"`)
    }

    switch (sig) {
      case 'continue':   return 'running'
      case 'need_input': return 'paused'
      case 'done':       return 'completed'
      case 'error':      return 'failed'
      case 'interrupt':  return 'interrupted'
      default:           throw new Error(`RunLifecycle: unknown signal "${sig as string}"`)
    }
  }
}
