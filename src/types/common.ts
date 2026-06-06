import type { ModelEvent } from './model.js'

export type JSONValue = string | number | boolean | null | JSONValue[] | { [k: string]: JSONValue }
export type JSONObject = Record<string, JSONValue>
export type JSONSchema = Record<string, unknown>

export interface Message {
  role: 'user' | 'assistant' | 'tool'
  content: MessageContent[]
}

export type MessageContent =
  | { type: 'text'; text: string }
  | { type: 'tool_use'; id: string; name: string; input: unknown }
  | { type: 'tool_result'; tool_use_id: string; content: string; is_error?: boolean }

export type TaskResult =
  | { status: 'success'; result: string }
  | { status: 'error'; reason: string; retryable?: boolean }
  | { status: 'interrupted'; checkpointId: string }

export interface AgentInvokeRequest {
  agentId: string
  goal: string
  input: string
  contextId?: string
  /** #82: per-turn variables injected into the turn-context region for this turn
   *  only (not persisted). Same shape #83 will reuse for persistent session vars. */
  variables?: Record<string, JSONValue>
  /** When provided, the run streams token-level ModelEvents to this callback. */
  onModelEvent?: (e: ModelEvent) => void
}

export interface ProjectionBound {
  /** Keep the newest N projections for the target context. Defaults to 5. */
  maxCount?: number
  /** Optional per-projection expiration, in seconds. */
  ttl?:      number
}

export interface ContextProjection {
  sourceRunId:     string
  sourceContextId?: string
  displayText:     string
  summary?:        string
  deliveredAt:     number
  attachedAt:      number
}

export interface AttachProjectionRequest {
  sourceRunId:      string
  sourceContextId?: string
  displayText:      string
  summary?:         string
  deliveredAt?:     number
  bound?:           ProjectionBound
}

export interface AgentResult {
  agentRunId:  string
  contextId:   string
  output:      string
  status:      'completed' | 'interrupted' | 'error'
  checkpointId?: string
}

export class InterruptSignal extends Error {
  constructor() {
    super('Agent interrupted')
    this.name = 'InterruptSignal'
  }
}

export class MaxIterationsError extends Error {
  constructor(state: string, max: number) {
    super(`State "${state}" exceeded max_iterations (${max})`)
    this.name = 'MaxIterationsError'
  }
}
