import type { Message } from './common.js'

export interface IStateStore {
  set(key: string, value: unknown, ttl?: number): Promise<void>
  get(key: string): Promise<unknown>
  delete(key: string): Promise<void>
  exists(key: string): Promise<boolean>
  /**
   * #83: enumerate entries whose key starts with `prefix` (expired entries skipped).
   * Needed for `listContextVars` and for reading all of a context's vars at invoke time.
   */
  list(prefix: string): Promise<Array<{ key: string; value: unknown }>>
}

export interface AgentCheckpoint {
  checkpointId: string
  sequence:     number
  goal:         string
  currentTurn?: string
  // #175 §8: schemaVersion >= 2 ⇒ v2 (lifecycle/userland, no fsm). Absent ⇒ v1 legacy.
  schemaVersion?: number
  // v1 legacy ONLY — the de-cored runtime no longer writes this; reads go through
  // readCheckpointLifecycle (runtime/checkpointSchema). Kept readable per D7.
  fsm?: {
    currentState: string
    resumeState?:  string
    stateData:    unknown
  }
  // #175 §8: v2 run-lifecycle. inline shape avoids a runtime→types cycle.
  lifecycle?: {
    status:       string
    resumeKind?:  'loop' | 'legacy-state'
  }
  // #175 §8: opaque, checkpointable userland blob (absent for the default loop).
  userland?: unknown
  context: {
    workingMemory:        unknown
    regions:              import('../context/Region.js').RegionSnapshot
    // Deprecated (kept readable for backwards compat parsing; AgentRuntime no
    // longer writes these. Will be removed when fixture format settles.):
    history?:             Message[]
    instructionsSnapshot?: string[]
    instructions?:        Record<string, string>
    contextEpoch?:        number
  }
  // #181: de-core removed the #60 pendingEvents queue; the runtime no longer
  // writes this. Optional so historical (v1) checkpoints that carry it still parse.
  pendingEvents?: AgentEvent[]
  children: ChildAgentRecord[]
  meta: {
    agentId:        string
    agentRunId:     string
    parentAgentId?: string
    timestamp:      number
    traceId:        string
    contextId?:     string
    activeSpanId?:  string
  }
}

export interface AgentEvent {
  type:    string
  payload: unknown
}

export interface ChildAgentRecord {
  taskId:        string
  agentId:       string
  runId?:        string
  contextId?:    string
  checkpointId?: string
  status:        'running' | 'success' | 'error' | 'interrupted'
}
