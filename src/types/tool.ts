import type { JSONSchema } from './common.js'
import type { WorkingMemory } from '../store/WorkingMemory.js'
import type { IStateStore } from './store.js'
import type { AgentFactory } from '../runtime/AgentFactory.js'
import type { GuardEvaluation, ObjectType, RelationType } from '../trace/types.js'

export interface ToolContext {
  workingMemory: WorkingMemory
  agentFactory:  AgentFactory
  stateStore:    IStateStore
  emit:          (event: string, payload?: unknown, guard?: GuardEvaluation | GuardEvaluation[]) => void
  requestSkill?: (name: string, scope?: 'turn' | 'session') => { requested: string; status: string; version?: string; scope?: 'turn' | 'session' }
  /**
   * #37: declare a content-addressable object (a passage read, a claim produced).
   * Returns a stable `objectId` handle (content-addressed → identical across
   * record/replay) the handler should surface in its result so the agent can later
   * cite it. Runtime records an `object.created` event after this tool's
   * `tool.responded`. Present only when the runtime wired a lineage sink.
   */
  createObject?:   (spec: { type: ObjectType; meta?: Record<string, unknown>; hash?: string }) => { objectId: string }
  /**
   * #38: declare a typed edge between two objects (e.g. a claim `cites` a passage).
   * Both ids must be objectIds minted by createObject. Runtime records a
   * `relation.created` event after `tool.responded`.
   */
  createRelation?: (spec: { type: RelationType; fromObjectId: string; toObjectId: string; meta?: Record<string, unknown> }) => { relationId: string }
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
