import { RunLifecycle } from '../runtime/RunLifecycle'

describe('RunLifecycle', () => {
  describe('construction', () => {
    it('starts in running', () => {
      expect(new RunLifecycle().state).toBe('running')
    })

    it('running is not terminal', () => {
      expect(new RunLifecycle().isTerminal()).toBe(false)
    })
  })

  describe('signals from running', () => {
    it('continue keeps it running', () => {
      const lc = new RunLifecycle()
      expect(lc.signal('continue')).toBe('running')
    })

    it('done completes the run (terminal)', () => {
      const lc = new RunLifecycle()
      expect(lc.signal('done')).toBe('completed')
      expect(lc.isTerminal()).toBe(true)
    })

    it('error fails the run (terminal)', () => {
      const lc = new RunLifecycle()
      expect(lc.signal('error')).toBe('failed')
      expect(lc.isTerminal()).toBe(true)
    })

    it('need_input pauses the run (suspended, not terminal)', () => {
      const lc = new RunLifecycle()
      expect(lc.signal('need_input')).toBe('paused')
      expect(lc.isTerminal()).toBe(false)
    })

    it('interrupt suspends to interrupted (not terminal)', () => {
      const lc = new RunLifecycle()
      expect(lc.signal('interrupt')).toBe('interrupted')
      expect(lc.isTerminal()).toBe(false)
    })
  })

  describe('resume (lifecycle re-entry — the kept half of transition())', () => {
    it('resumes a paused run back to running', () => {
      const lc = new RunLifecycle()
      lc.signal('need_input')
      expect(lc.signal('resume')).toBe('running')
    })

    it('resumes an interrupted run back to running', () => {
      const lc = new RunLifecycle()
      lc.signal('interrupt')
      expect(lc.signal('resume')).toBe('running')
    })
  })

  describe('illegal transitions are rejected (state defs stay sound)', () => {
    it('a terminal completed run rejects further signals', () => {
      const lc = new RunLifecycle()
      lc.signal('done')
      expect(() => lc.signal('continue')).toThrow(/terminal|completed/i)
    })

    it('a terminal failed run rejects further signals', () => {
      const lc = new RunLifecycle()
      lc.signal('error')
      expect(() => lc.signal('resume')).toThrow(/terminal|failed/i)
    })

    it('resume is illegal when not suspended', () => {
      const lc = new RunLifecycle()
      expect(() => lc.signal('resume')).toThrow(/resume|running|suspend/i)
    })
  })

  describe('snapshot / restore (cross-turn resume via checkpoint)', () => {
    it('snapshot captures the current state', () => {
      const lc = new RunLifecycle()
      lc.signal('need_input')
      expect(lc.snapshot()).toEqual({ state: 'paused' })
    })

    it('restore reconstructs a suspended run that can then resume', () => {
      const lc = RunLifecycle.restore({ state: 'paused' })
      expect(lc.state).toBe('paused')
      expect(lc.signal('resume')).toBe('running')
    })

    it('restore of a terminal run stays terminal', () => {
      const lc = RunLifecycle.restore({ state: 'completed' })
      expect(lc.isTerminal()).toBe(true)
    })

    it('restore rejects an unknown state', () => {
      expect(() => RunLifecycle.restore({ state: 'bogus' as never })).toThrow(/unknown|invalid|bogus/i)
    })
  })
})
