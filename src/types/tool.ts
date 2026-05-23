import type { JSONSchema } from './common.js'
import type { WorkingMemory } from '../store/WorkingMemory.js'
import type { IStateStore } from './store.js'
import type { AgentFactory } from '../runtime/AgentFactory.js'

export interface ToolContext {
  workingMemory: WorkingMemory
  agentFactory:  AgentFactory
  stateStore:    IStateStore
  emit:          (event: string, payload?: unknown) => void
  requestSkill?: (name: string) => { requested: string; status: string; version?: string }
}

export interface ToolDefinition {
  name:          string
  description:   string
  inputSchema:   JSONSchema
  handler:       (input: unknown, ctx: ToolContext) => Promise<unknown>
  parallelSafe?: boolean
}

export interface ToolCall {
  id:    string
  name:  string
  input: unknown
}

export interface ToolResult {
  toolCallId: string
  toolName:   string
  output:     unknown
  error?:     string
  isError:    boolean
  duration:   number
}
