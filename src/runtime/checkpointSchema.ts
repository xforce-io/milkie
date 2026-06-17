import type { AgentCheckpoint } from '../types/store.js'

/**
 * Checkpoint schema versioning (#175 §8 / D7).
 *
 * The de-core removes the multi-state FSM, so the old `checkpoint.fsm`
 * `{currentState, resumeState}` surface no longer reflects how runs resume: a
 * single-state agent's FSM is always at `states[0]`, and run-lifecycle is owned
 * by `RunLifecycle`. The new runtime writes a versioned v2 checkpoint with an
 * explicit `lifecycle` (and an opaque `userland` blob for future upper-SMs); it
 * never writes `fsm`. Reads must still accept v1 (legacy `{fsm}`) so existing
 * event logs / portable sessions / serve restart recovery keep working.
 */
export const CHECKPOINT_SCHEMA_VERSION = 2

export interface CheckpointLifecycle {
  /** RunLifecycle status (running/paused/interrupted/completed/failed). */
  status: string
  /**
   * How to resume: 'loop' re-enters the single user state and continues the
   * autonomous loop; 'legacy-state' is a v1 checkpoint whose old `resumeState`
   * we deliberately do NOT expose as a business state — it also re-enters the
   * default loop.
   */
  resumeKind: 'loop' | 'legacy-state'
  /** An interrupt/need_input-style resumable pause (vs running/terminal). */
  suspended: boolean
}

/**
 * Read a checkpoint's run-lifecycle, supporting v2 ({schemaVersion>=2,
 * lifecycle}) and v1 legacy ({fsm}). The new runtime never reads `fsm` for
 * control flow — this is the single v1↔v2 compatibility seam.
 */
export function readCheckpointLifecycle(cp: AgentCheckpoint): CheckpointLifecycle {
  if ((cp.schemaVersion ?? 0) >= 2 && cp.lifecycle) {
    const status = cp.lifecycle.status
    return {
      status,
      resumeKind: cp.lifecycle.resumeKind ?? 'loop',
      suspended:  status === 'paused' || status === 'interrupted',
    }
  }
  // v1 legacy: map the old reserved FSM state. We do NOT resurrect the business
  // topology — `resumeState` is ignored; resume always re-enters the loop.
  const fsm = cp.fsm
  if (fsm) {
    if (fsm.currentState === 'paused' && fsm.resumeState) {
      return { status: 'paused', resumeKind: 'legacy-state', suspended: true }
    }
    if (fsm.currentState === 'failed') {
      return { status: 'failed', resumeKind: 'legacy-state', suspended: false }
    }
    return { status: 'running', resumeKind: 'legacy-state', suspended: false }
  }
  return { status: 'running', resumeKind: 'loop', suspended: false }
}
