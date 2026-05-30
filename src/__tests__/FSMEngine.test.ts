import { FSMEngine } from '../fsm/FSMEngine'
import type { FSMDefinition } from '../types/agent'

const reactFSM: FSMDefinition = {
  states: [
    { name: 'react', type: 'llm', tools: ['search', 'calculate'] },
  ],
}

const routingFSM: FSMDefinition = {
  states: [
    {
      name:  'classify',
      type:  'llm',
      tools: ['classify_intent'],
      on:    { INTENT_A: 'handle_a', INTENT_B: 'handle_b' },
    },
    {
      name:    'handle_a',
      type:    'action',
      handler: 'doA',
      on:      { DONE: 'classify' },
    },
    {
      name:    'handle_b',
      type:    'action',
      handler: 'doB',
      on:      { DONE: 'classify' },
    },
  ],
}

const terminalFSM: FSMDefinition = {
  states: [
    { name: 'start', type: 'llm', on: { DONE: 'end' } },
    { name: 'end',   type: 'action', terminal: true },
  ],
}

describe('FSMEngine', () => {
  describe('initial state', () => {
    it('starts at first declared state', () => {
      const fsm = new FSMEngine(reactFSM)
      expect(fsm.currentStateName).toBe('react')
    })

    it('throws on empty states', () => {
      expect(() => new FSMEngine({ states: [] })).toThrow()
    })
  })

  describe('ctx.emit() / processPendingEvent()', () => {
    it('transitions on emitted event', () => {
      const fsm = new FSMEngine(routingFSM)
      fsm.emitEvent('INTENT_A')
      const next = fsm.processPendingEvent()
      expect(next?.name).toBe('handle_a')
      expect(fsm.currentStateName).toBe('handle_a')
    })

    it('first event wins (second emit ignored)', () => {
      const fsm = new FSMEngine(routingFSM)
      fsm.emitEvent('INTENT_A')
      fsm.emitEvent('INTENT_B')
      fsm.processPendingEvent()
      expect(fsm.currentStateName).toBe('handle_a')
    })

    it('returns null when no pending event', () => {
      const fsm = new FSMEngine(routingFSM)
      expect(fsm.processPendingEvent()).toBeNull()
    })

    it('ignores events with no matching on: entry', () => {
      const fsm = new FSMEngine(routingFSM)
      fsm.emitEvent('UNKNOWN_EVENT')
      const next = fsm.processPendingEvent()
      expect(next).toBeNull()
      expect(fsm.currentStateName).toBe('classify')
    })

    it('carries guard evaluations through to the onTransition callback', () => {
      const fsm = new FSMEngine(routingFSM)   // classify, on INTENT_A: handle_a
      let received: import('../fsm/FSMEngine').FSMEvent | undefined
      fsm.onTransitionCallback((_from, _to, event) => { received = event })

      fsm.emitEvent('INTENT_A', undefined, {
        guardId: 'g', result: 'INTENT_A', contextSlice: { s: 1 },
      })
      fsm.processPendingEvent()

      expect(received?.guard).toEqual([
        { guardId: 'g', result: 'INTENT_A', contextSlice: { s: 1 } },
      ])
    })

    it('normalizes a guard array argument as-is', () => {
      const fsm = new FSMEngine(routingFSM)
      let received: import('../fsm/FSMEngine').FSMEvent | undefined
      fsm.onTransitionCallback((_f, _t, e) => { received = e })
      const arr = [{ guardId: 'a', result: 1, contextSlice: {} }, { guardId: 'b', result: 2, contextSlice: {} }]
      fsm.emitEvent('INTENT_A', undefined, arr)
      fsm.processPendingEvent()
      expect(received?.guard).toEqual(arr)
    })
  })

  describe('processDone()', () => {
    it('transitions via on.DONE', () => {
      const fsm = new FSMEngine(terminalFSM)
      const next = fsm.processDone()
      expect(next?.name).toBe('end')
      expect(fsm.currentStateName).toBe('end')
    })

    it('returns null when no on.DONE defined', () => {
      const fsm = new FSMEngine(reactFSM)
      expect(fsm.processDone()).toBeNull()
      expect(fsm.currentStateName).toBe('react')
    })
  })

  describe('terminal states', () => {
    it('isTerminal() returns true for terminal states', () => {
      const fsm = new FSMEngine(terminalFSM)
      fsm.processDone()
      expect(fsm.isTerminal()).toBe(true)
    })

    it('isTerminal() returns false for non-terminal states', () => {
      const fsm = new FSMEngine(reactFSM)
      expect(fsm.isTerminal()).toBe(false)
    })
  })

  describe('global transitions', () => {
    it('interrupt event transitions to paused regardless of state', () => {
      const fsm = new FSMEngine(routingFSM)
      fsm.emitEvent('interrupt')
      const next = fsm.processPendingEvent()
      expect(next?.name).toBe('paused')
      expect(fsm.currentStateName).toBe('paused')
    })

    it('error event transitions to error_handling', () => {
      const fsm = new FSMEngine(routingFSM)
      fsm.emitEvent('error')
      const next = fsm.processPendingEvent()
      expect(next?.name).toBe('error_handling')
    })
  })

  describe('transition callbacks', () => {
    it('fires callback on each transition', () => {
      const fsm = new FSMEngine(routingFSM)
      const transitions: Array<[string, string]> = []
      fsm.onTransitionCallback((from, to) => transitions.push([from, to]))

      fsm.emitEvent('INTENT_A')
      fsm.processPendingEvent()

      expect(transitions).toEqual([['classify', 'handle_a']])
    })
  })

  describe('snapshot / restore', () => {
    it('restores state from snapshot', () => {
      const fsm = new FSMEngine(routingFSM)
      fsm.emitEvent('INTENT_A')
      fsm.processPendingEvent()

      const snap = fsm.snapshot()
      expect(snap.currentState).toBe('handle_a')

      const fsm2 = new FSMEngine(routingFSM)
      fsm2.restore(snap)
      expect(fsm2.currentStateName).toBe('handle_a')
    })
  })
})
