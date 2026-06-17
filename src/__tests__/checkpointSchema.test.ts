import { readCheckpointLifecycle, CHECKPOINT_SCHEMA_VERSION } from '../runtime/checkpointSchema'
import type { AgentCheckpoint } from '../types/store'

// #175 §8/D7: the v1↔v2 checkpoint compatibility seam. New runtime writes v2
// ({schemaVersion, lifecycle}, no fsm); reads must still accept v1 ({fsm}).
const cp = (partial: Partial<AgentCheckpoint>): AgentCheckpoint => partial as AgentCheckpoint

describe('readCheckpointLifecycle (v1/v2 seam)', () => {
  describe('v2 (schemaVersion >= 2, lifecycle)', () => {
    it('reads an interrupted v2 checkpoint as suspended/loop', () => {
      expect(readCheckpointLifecycle(cp({
        schemaVersion: CHECKPOINT_SCHEMA_VERSION,
        lifecycle: { status: 'interrupted', resumeKind: 'loop' },
      }))).toEqual({ status: 'interrupted', resumeKind: 'loop', suspended: true })
    })

    it('reads a paused (need_input) v2 checkpoint as suspended', () => {
      expect(readCheckpointLifecycle(cp({
        schemaVersion: 2, lifecycle: { status: 'paused', resumeKind: 'loop' },
      })).suspended).toBe(true)
    })

    it('reads a running v2 checkpoint as not suspended; resumeKind defaults to loop', () => {
      expect(readCheckpointLifecycle(cp({
        schemaVersion: 2, lifecycle: { status: 'running' },
      }))).toEqual({ status: 'running', resumeKind: 'loop', suspended: false })
    })

    it('prefers v2 lifecycle even if a legacy fsm field is also present', () => {
      expect(readCheckpointLifecycle(cp({
        schemaVersion: 2,
        lifecycle: { status: 'interrupted', resumeKind: 'loop' },
        fsm: { currentState: 'react', stateData: null },
      })).status).toBe('interrupted')
    })
  })

  describe('v1 legacy (fsm, no schemaVersion)', () => {
    it('maps a paused fsm checkpoint to suspended/legacy-state (resumeState NOT exposed)', () => {
      expect(readCheckpointLifecycle(cp({
        fsm: { currentState: 'paused', resumeState: 'react', stateData: null },
      }))).toEqual({ status: 'paused', resumeKind: 'legacy-state', suspended: true })
    })

    it('maps a non-paused continuation fsm checkpoint to running (resumable, not suspended)', () => {
      expect(readCheckpointLifecycle(cp({
        fsm: { currentState: 'react', stateData: null },
      }))).toEqual({ status: 'running', resumeKind: 'legacy-state', suspended: false })
    })

    it('maps a failed fsm checkpoint to failed/not-suspended', () => {
      expect(readCheckpointLifecycle(cp({
        fsm: { currentState: 'failed', stateData: null },
      })).suspended).toBe(false)
    })
  })

  it('defaults to running/loop when neither lifecycle nor fsm is present', () => {
    expect(readCheckpointLifecycle(cp({}))).toEqual({ status: 'running', resumeKind: 'loop', suspended: false })
  })
})
