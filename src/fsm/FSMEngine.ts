import type { FSMDefinition, FSMState } from '../types/agent.js'
import type { FsmEventDomain } from '../trace/types.js'

export interface FSMEvent {
  name:     string
  payload?: unknown
  /**
   * Event domain — set at emit site so trace consumers can tell apart
   * framework lifecycle / global signals / runtime control flow / business
   * (tool-emitted) events. When omitted, treated as 'business' since the
   * only path that doesn't set it is `ctx.emit()` from tool handlers.
   */
  domain?:  FsmEventDomain
}

export type FSMTransitionHandler = (from: string, to: string, event: FSMEvent) => void

// FSM reserved states injected by the framework (not declared in user config)
const RESERVED_STATES = ['error_handling', 'paused', 'failed'] as const

export class FSMEngine {
  private current:     FSMState
  private states:      Map<string, FSMState>
  private pendingEvent: FSMEvent | null = null
  private onTransition: FSMTransitionHandler | null = null

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

  onTransitionCallback(fn: FSMTransitionHandler): void {
    this.onTransition = fn
  }

  // Called from tool handlers via ctx.emit() and from AgentRuntime for
  // interrupt/error signals. Default domain is 'business' — the global
  // 'interrupt'/'error' branches in processPendingEvent re-tag those to
  // 'signal' on the transition itself, so this default is safe.
  emitEvent(event: string, payload?: unknown): void {
    if (this.pendingEvent) {
      // First event wins within a single tool execution
      return
    }
    this.pendingEvent = { name: event, payload, domain: 'business' }
  }

  // Process the pending event (if any) and transition to the next state.
  // Returns the new state name, or null if no transition occurred.
  processPendingEvent(): FSMState | null {
    const event = this.pendingEvent
    this.pendingEvent = null

    if (!event) return null

    // Global transitions override state-specific ones. Stamp the domain as
    // 'signal' for these forced jumps; tool-emitted 'interrupt'/'error' names
    // arrive without a domain and we re-tag them here at the global handler.
    if (event.name === 'interrupt') {
      return this.transition('paused', { ...event, domain: 'signal' })
    }
    if (event.name === 'error') {
      return this.transition('error_handling', { ...event, domain: 'signal' })
    }

    const target = this.current.on?.[event.name]
    if (!target) {
      // No transition defined for this event — ignore
      return null
    }
    return this.transition(target, event)
  }

  // Process the DONE event (LLM produced text output)
  processDone(): FSMState | null {
    const target = this.current.on?.['DONE']
    if (!target) return null
    return this.transition(target, { name: 'DONE', domain: 'lifecycle' })
  }

  transitionTo(stateName: string, event?: FSMEvent): FSMState {
    return this.transition(stateName, event ?? { name: 'DIRECT', domain: 'lifecycle' })
  }

  private transition(stateName: string, event: FSMEvent): FSMState {
    const from = this.current.name

    // Allow reserved states without being in the config
    if (RESERVED_STATES.includes(stateName as typeof RESERVED_STATES[number])) {
      const reserved: FSMState = {
        name:     stateName,
        type:     'action',
        terminal: stateName === 'paused' || stateName === 'failed',
      }
      this.onTransition?.(from, stateName, event)
      this.current = reserved
      return reserved
    }

    const next = this.states.get(stateName)
    if (!next) {
      throw new Error(`FSM: transition to unknown state "${stateName}" from "${from}" via "${event.name}"`)
    }

    this.onTransition?.(from, stateName, event)
    this.current = next
    return next
  }

  getState(name: string): FSMState | undefined {
    return this.states.get(name)
  }

  hasPendingEvent(): boolean {
    return this.pendingEvent !== null
  }

  snapshot(resumeState?: string): { currentState: string; resumeState?: string; stateData: unknown } {
    return { currentState: this.current.name, resumeState, stateData: null }
  }

  restore(snapshot: { currentState: string }): void {
    if (RESERVED_STATES.includes(snapshot.currentState as typeof RESERVED_STATES[number])) {
      this.current = {
        name:     snapshot.currentState,
        type:     'action',
        terminal: snapshot.currentState === 'paused' || snapshot.currentState === 'failed',
      }
      return
    }

    const state = this.states.get(snapshot.currentState)
    if (!state) {
      throw new Error(`FSM restore: unknown state "${snapshot.currentState}"`)
    }
    this.current = state
  }
}
