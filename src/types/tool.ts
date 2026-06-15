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
  /** #164: raw user input for the current turn, stable across all tool-loop iterations. */
  currentTurn?:  string
  /**
   * #37: declare a content-addressable object (a passage read, a claim produced).
   * Returns a stable `objectId` handle (content-addressed → identical across
   * record/replay) the handler should surface in its result so the agent can later
   * cite it. Runtime records an `object.created` event after this tool's
   * `tool.responded`. Present only when the runtime wired a lineage sink.
   */
  createObject?:   (spec: { type: ObjectType; meta?: Record<string, unknown>; hash?: string }) => { objectId: string }
  /**
   * #113 P2 (lazy): register a candidate object (e.g. a wide-recall grep hit)
   * WITHOUT emitting object.created. Returns its content-addressed objectId so the
   * agent can cite it; it only becomes an event if a later cite promotes it. Keeps
   * large recall from flooding the event log with never-cited candidates.
   */
  registerObject?: (spec: { type: ObjectType; meta?: Record<string, unknown>; hash?: string }) => { objectId: string }
  /**
   * #113 P2: emit object.created for a previously registered (lazy) object, the
   * first time it is cited. Idempotent and a no-op for already-emitted objects.
   */
  promoteObject?:  (objectId: string) => void
  /**
   * #38: declare a typed edge between two objects (e.g. a claim `cites` a passage).
   * Both ids must be objectIds minted by createObject. Runtime records a
   * `relation.created` event after `tool.responded`.
   */
  createRelation?: (spec: { type: RelationType; fromObjectId: string; toObjectId: string; meta?: Record<string, unknown> }) => { relationId: string }
  /**
   * #113 P1: resolve an objectId minted earlier this run via createObject
   * (read/grep), or undefined if it was never minted. Lets a producer fail-fast on
   * a fabricated/hallucinated id — e.g. cite rejecting an objectId the agent invented
   * — and underpins lazy-promote (P2). Present only when the runtime wired lineage.
   */
  resolveObject?:  (objectId: string) => { type: ObjectType; meta?: Record<string, unknown> } | undefined
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
