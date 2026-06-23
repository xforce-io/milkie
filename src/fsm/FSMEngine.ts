import type { FSMDefinition, FSMState } from '../types/agent.js'

/**
 * FSMEngine — post #175 de-core: a SINGLE-STATE execution holder plus the
 * framework's reserved lifecycle states. The "developer-authored multi-state
 * business FSM" (intent routing / DialogFlow / action chains) is gone:
 *
 *   - No `on:` map jumps, no `pendingEvent`/`emitEvent`/`processPendingEvent`,
 *     no `processDone` business transitions. Business topology lives in userland
 *     now (slot-filling + precondition; see docs/design/175 §5/§6).
 *   - What survives is the lower run-lifecycle re-entry (§5 kept rows): the
 *     reserved states `paused` / `error_handling` / `failed`, the suspend/resume
 *     re-entry (`X → paused`, `paused → X` via RESUME), and the self-loop
 *     (`X → X`, i.e. the loop keeps running). The single user state is the only
 *     authored state; everything else is framework-fixed.
 *
 * Run-status authority is `RunLifecycle`; this engine only tracks WHICH state
 * the loop body currently sits in so reserved re-entry (interrupt/retry/resume)
 * can land back on it.
 */
export interface FSMEvent {
  name:     string
  payload?: unknown
}

// FSM reserved states injected by the framework (not declared in user config)
const RESERVED_STATES = ['error_handling', 'paused', 'failed'] as const

export class FSMEngine {
  private current: FSMState
  private states:  Map<string, FSMState>

  constructor(definition: FSMDefinition) {
    if (definition.states.length === 0) {
      throw new Error('FSM must have at least one state')
    }
    this.states = new Map(definition.states.map(s => [s.name, s]))
    this.current = definition.states[0]!
  }

  get currentState(): FSMState {
    return this.current
  }

  get currentStateName(): string {
    return this.current.name
  }

  isTerminal(): boolean {
    return this.current.terminal === true
  }

  /**
   * Is the current state a framework-reserved terminal lifecycle state
   * (`paused` / `failed`)? These persist their own checkpoint at the
   * suspend/fail point, so the run() success path must NOT re-write one.
   * Distinct from a user-defined terminal state (e.g. `completed`), whose
   * final working memory still needs a continuation checkpoint (#172), and
   * robust against a user state that merely happens to be *named* `paused`
   * but is non-terminal (the terminal guard keeps it out of this set).
   */
  isReservedTerminal(): boolean {
    return this.isTerminal()
      && RESERVED_STATES.includes(this.current.name as typeof RESERVED_STATES[number])
  }

  /**
   * Re-enter a state (reserved lifecycle state or the run's single user state).
   * Used for suspend (`X → paused`), error escalation (`X → error_handling`),
   * retry-back, and resume (`paused → X`). NOT a business transition — there is
   * no longer an `on:` table; the only user-state target is the one the run
   * already holds (resume) or its prior self.
   */
  transitionTo(stateName: string, _event?: FSMEvent): FSMState {
    if (RESERVED_STATES.includes(stateName as typeof RESERVED_STATES[number])) {
      const reserved: FSMState = {
        name:     stateName,
        type:     'action',
        terminal: stateName === 'paused' || stateName === 'failed',
      }
      this.current = reserved
      return reserved
    }

    const next = this.states.get(stateName)
    if (!next) {
      throw new Error(`FSM: re-entry to unknown state "${stateName}" from "${this.current.name}"`)
    }
    this.current = next
    return next
  }

  getState(name: string): FSMState | undefined {
    return this.states.get(name)
  }
}
