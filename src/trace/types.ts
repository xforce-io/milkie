import type { ModelRequest, ModelResponse } from '../types/model.js'

/**
 * Agent Trace event types.
 *
 * Phase 3 adds lifecycle events and content-addressed cache fields on I/O
 * payloads. Spawn/fork events are deferred to Phase 5.
 */
export type EventKind =
  | 'llm.requested'
  | 'llm.responded'
  | 'tool.requested'
  | 'tool.responded'
  | 'agent.run.started'
  | 'agent.run.completed'

export interface Event<P = unknown> {
  id: string
  runId: string
  type: EventKind
  actor: string
  causedBy?: string
  timestamp: number
  payload: P
}

// ---- I/O payloads (Phase 2 shapes + Phase 3 requestHash) ----

export interface LlmRequestedPayload {
  request: ModelRequest
  /** Phase 3: hash of canonicalized request; cache key for replay. */
  requestHash: string
}

export interface LlmRespondedPayload {
  response: ModelResponse
  /** Mirrors the requested-event hash so consumers don't need to re-join. */
  requestHash: string
}

export interface ToolRequestedPayload {
  toolName: string
  input: unknown
  /** Phase 3: hash of canonicalized (toolName + input); cache key for replay. */
  requestHash: string
}

export interface ToolRespondedPayload {
  toolName: string
  output?: unknown
  /** Phase 3: structured to preserve retryable/code/name; replay rebuilds Error. */
  error?: {
    message:    string
    retryable?: boolean
    code?:      string
    name?:      string
  }
  /** Mirrors the requested-event hash. */
  requestHash: string
}

// ---- Lifecycle payloads (Phase 3) ----

export interface AgentRunStartedPayload {
  agentId:    string
  goal:       string
  input:      string
  contextId:  string
  parentId?:  string
}

export interface AgentRunCompletedPayload {
  status:           'completed' | 'interrupted' | 'error'
  lastTextOutput?:  string
  error?:           string
}

// ---- Typed event aliases ----

export type LlmRequestedEvent       = Event<LlmRequestedPayload>       & { type: 'llm.requested' }
export type LlmRespondedEvent       = Event<LlmRespondedPayload>       & { type: 'llm.responded' }
export type ToolRequestedEvent      = Event<ToolRequestedPayload>      & { type: 'tool.requested' }
export type ToolRespondedEvent      = Event<ToolRespondedPayload>      & { type: 'tool.responded' }
export type AgentRunStartedEvent    = Event<AgentRunStartedPayload>    & { type: 'agent.run.started' }
export type AgentRunCompletedEvent  = Event<AgentRunCompletedPayload>  & { type: 'agent.run.completed' }

export type AnyEvent =
  | LlmRequestedEvent
  | LlmRespondedEvent
  | ToolRequestedEvent
  | ToolRespondedEvent
  | AgentRunStartedEvent
  | AgentRunCompletedEvent
