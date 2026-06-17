import { FSMEngine } from '../fsm/FSMEngine'
import type { FSMDefinition } from '../types/agent'

// #175 de-core: FSMEngine is now a SINGLE-STATE holder. The developer-authored
// multi-state business FSM (on:/emit/processPendingEvent/processDone/guard) is
// gone; what survives is the run's single user state plus the framework's
// reserved lifecycle re-entry (paused / error_handling / failed).

const reactFSM: FSMDefinition = {
  states: [
    { name: 'react', type: 'llm', tools: ['search', 'calculate'] },
  ],
}

const terminalFSM: FSMDefinition = {
  states: [
    { name: 'start', type: 'llm' },
    { name: 'end',   type: 'action', terminal: true },
  ],
}

describe('FSMEngine', () => {
  describe('initial state', () => {
    it('starts at first declared state', () => {
      const fsm = new FSMEngine(reactFSM)
      expect(fsm.currentStateName).toBe('react')
    })

    it('exposes the current state object via currentState', () => {
      const fsm = new FSMEngine(reactFSM)
      expect(fsm.currentState.name).toBe('react')
      expect(fsm.currentState.type).toBe('llm')
    })

    it('throws on empty states', () => {
      expect(() => new FSMEngine({ states: [] })).toThrow()
    })
  })

  describe('getState', () => {
    it('returns a declared state by name', () => {
      const fsm = new FSMEngine(terminalFSM)
      expect(fsm.getState('end')?.terminal).toBe(true)
    })

    it('returns undefined for an unknown state', () => {
      const fsm = new FSMEngine(terminalFSM)
      expect(fsm.getState('nope')).toBeUndefined()
    })
  })

  describe('terminal states', () => {
    it('isTerminal() returns false for the initial non-terminal user state', () => {
      const fsm = new FSMEngine(reactFSM)
      expect(fsm.isTerminal()).toBe(false)
    })

    it('isTerminal() reflects the terminal flag after a re-entry', () => {
      const fsm = new FSMEngine(terminalFSM)
      expect(fsm.isTerminal()).toBe(false)
      fsm.transitionTo('end')   // re-entry to the run's terminal user state
      expect(fsm.isTerminal()).toBe(true)
    })
  })

  describe('reserved lifecycle re-entry (transitionTo)', () => {
    it('re-enters the reserved `paused` state (suspend) which is terminal', () => {
      const fsm = new FSMEngine(reactFSM)
      const paused = fsm.transitionTo('paused')
      expect(paused.name).toBe('paused')
      expect(fsm.currentStateName).toBe('paused')
      expect(fsm.isTerminal()).toBe(true)
    })

    it('re-enters the reserved `error_handling` state (escalation), non-terminal', () => {
      const fsm = new FSMEngine(reactFSM)
      const err = fsm.transitionTo('error_handling')
      expect(err.name).toBe('error_handling')
      expect(fsm.currentStateName).toBe('error_handling')
      expect(fsm.isTerminal()).toBe(false)
    })

    it('re-enters the reserved `failed` state which is terminal', () => {
      const fsm = new FSMEngine(reactFSM)
      fsm.transitionTo('failed')
      expect(fsm.currentStateName).toBe('failed')
      expect(fsm.isTerminal()).toBe(true)
    })

    it('resumes from `paused` back to the run user state (paused → X via RESUME)', () => {
      const fsm = new FSMEngine(reactFSM)
      fsm.transitionTo('paused')
      const resumed = fsm.transitionTo('react')
      expect(resumed.name).toBe('react')
      expect(fsm.currentStateName).toBe('react')
      expect(fsm.isTerminal()).toBe(false)
    })

    it('self-loop re-entry keeps the run on its single user state', () => {
      const fsm = new FSMEngine(reactFSM)
      fsm.transitionTo('react')
      expect(fsm.currentStateName).toBe('react')
    })

    it('throws when re-entering an unknown (non-reserved, non-declared) state', () => {
      const fsm = new FSMEngine(reactFSM)
      expect(() => fsm.transitionTo('handle_a')).toThrow(/unknown state/)
    })
  })

  describe('snapshot / restore', () => {
    it('snapshots the current user state', () => {
      const fsm = new FSMEngine(reactFSM)
      const snap = fsm.snapshot()
      expect(snap.currentState).toBe('react')
    })

    it('carries the optional resumeState through the snapshot', () => {
      const fsm = new FSMEngine(reactFSM)
      fsm.transitionTo('paused')
      const snap = fsm.snapshot('react')
      expect(snap.currentState).toBe('paused')
      expect(snap.resumeState).toBe('react')
    })

    it('restores a user state from a snapshot', () => {
      const fsm = new FSMEngine(terminalFSM)
      fsm.transitionTo('end')
      const snap = fsm.snapshot()
      expect(snap.currentState).toBe('end')

      const fsm2 = new FSMEngine(terminalFSM)
      fsm2.restore(snap)
      expect(fsm2.currentStateName).toBe('end')
      expect(fsm2.isTerminal()).toBe(true)
    })

    it('restores a reserved lifecycle state from a snapshot', () => {
      const fsm = new FSMEngine(reactFSM)
      fsm.transitionTo('paused')
      const snap = fsm.snapshot('react')

      const fsm2 = new FSMEngine(reactFSM)
      fsm2.restore(snap)
      expect(fsm2.currentStateName).toBe('paused')
      expect(fsm2.isTerminal()).toBe(true)
    })

    it('throws when restoring an unknown state', () => {
      const fsm = new FSMEngine(reactFSM)
      expect(() => fsm.restore({ currentState: 'ghost' })).toThrow(/unknown state/)
    })
  })
})
