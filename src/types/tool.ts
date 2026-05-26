import type { JSONSchema } from './common.js'
import type { WorkingMemory } from '../store/WorkingMemory.js'
import type { IStateStore } from './store.js'
import type { AgentFactory } from '../runtime/AgentFactory.js'

export interface ToolContext {
  workingMemory: WorkingMemory
  agentFactory:  AgentFactory
  stateStore:    IStateStore
  emit:          (event: string, payload?: unknown) => void
  requestSkill?: (name: string, scope?: 'turn' | 'session') => { requested: string; status: string; version?: string; scope?: 'turn' | 'session' }
}

export interface ToolDefinition {
  name:            string
  description:     string
  inputSchema:     JSONSchema
  handler:         (input: unknown, ctx: ToolContext) => Promise<unknown>
  parallelSafe?:   boolean
  /** PR-E Phase 1: how raw tool output is shaped before entering LLM context. Default verbatim. */
  resultStrategy?: ToolResultStrategy
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

// ---- PR-E Phase 1: Tool result shaping ----
// Spec: docs/superpowers/specs/2026-05-25-context-region-substrate-design.md §4.4
//
// Phase 1 ships shape + onError axes only. visibility / target stay implicit
// ('inline' / 'scratchpad') — those need context_fetch tool + tool_use/result
// pairing handling, deferred to Phase 2.

export type Shape =
  | 'verbatim'
  | { kind: 'truncate'; maxChars: number; tailHint?: boolean }
  | { kind: 'tail';     maxChars: number }

export interface ToolResultStrategy {
  shape:    Shape
  /** Shape applied when the tool handler throws. Default 'verbatim' (full error info to agent). */
  onError?: Shape
}
