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
  fsm: {
    currentState: string
    resumeState?:  string
    stateData:    unknown
  }
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
  pendingEvents: AgentEvent[]
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
