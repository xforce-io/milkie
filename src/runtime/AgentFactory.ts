import type { AgentConfig, BuiltinToolName } from '../types/agent.js'
import type { AgentInvokeRequest, AgentResult, RunControlOptions } from '../types/common.js'
import type { IStateStore } from '../types/store.js'
import type { ITrajectoryRecorder } from '../types/trajectory.js'
import type { ToolDefinition } from '../types/tool.js'
import type { IIOPort } from './IOPort.js'
import type { RunControl } from './RunControl.js'

export interface AgentSpawnOptions {
  config:      AgentConfig
  goal:        string
  input:       string
  contextId?:  string
  agentRunId?: string
  parentId?:   string
  stateStore:  IStateStore
  recorder:    ITrajectoryRecorder
  ioPort:      IIOPort
  causalCursor?:  import('../trace/CausalCursor.js').CausalCursor
  extraTools?:    ToolDefinition[]
  eventStore?:    import('../trace/EventStore.js').IEventStore
  makeChildPort?: import('./AgentRuntime.js').MakeChildPort
  /** #235: parent effective built-in allowlist; child can only narrow. */
  parentBuiltinTools?: readonly BuiltinToolName[]
  /** #237: child-local control options (still bounded by parentRunControl). */
  control?: RunControlOptions
  /** #237: parent run control inheritance. */
  parentRunControl?: RunControl
}

// Forward declaration to avoid circular import — resolved at runtime
export type SpawnFn = (opts: AgentSpawnOptions) => Promise<AgentResult>

export class AgentFactory {
  constructor(private readonly spawnFn: SpawnFn) {}

  async spawn(opts: AgentSpawnOptions): Promise<AgentResult> {
    return this.spawnFn(opts)
  }
}
