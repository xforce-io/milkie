import type { AgentConfig } from '../types/agent.js'
import type { AgentInvokeRequest, AgentResult } from '../types/common.js'
import type { IStateStore } from '../types/store.js'
import type { ITrajectoryRecorder } from '../types/trajectory.js'
import type { ToolDefinition } from '../types/tool.js'
import type { IIOPort } from './IOPort.js'

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
  extraTools?: ToolDefinition[]
}

// Forward declaration to avoid circular import — resolved at runtime
export type SpawnFn = (opts: AgentSpawnOptions) => Promise<AgentResult>

export class AgentFactory {
  constructor(private readonly spawnFn: SpawnFn) {}

  async spawn(opts: AgentSpawnOptions): Promise<AgentResult> {
    return this.spawnFn(opts)
  }
}
