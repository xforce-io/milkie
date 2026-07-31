import type { InvalidToolArguments, Message, MessageContent } from './common.js'
import type { ToolCall } from './tool.js'
import type { JSONSchema } from './common.js'

export type ModelErrorCode =
  | 'MODEL_CONNECTION_ERROR'
  | 'MODEL_TIMEOUT'
  | 'MODEL_RATE_LIMITED'
  | 'MODEL_AUTH_ERROR'
  | 'MODEL_BAD_RESPONSE'
  | 'MODEL_UNKNOWN_ERROR'

export type ModelErrorPhase = 'request' | 'stream_open' | 'stream_read' | 'response_parse'

export interface ModelErrorEnvelope {
  code:       ModelErrorCode
  message:    string
  phase:      ModelErrorPhase
  provider:   string
  model:      string
  retryable:  boolean
  status?:    number
}

export interface MaxIterationsErrorEnvelope {
  code:       'MAX_ITERATIONS_EXCEEDED'
  message:    string
  phase:      'agent_loop'
  retryable:  false
  provider?:  undefined
  model?:     undefined
}

export interface AbandonedRunErrorEnvelope {
  code:       'RUN_ABANDONED'
  message:    string
  phase:      'recovery'
  retryable:  true
  provider?:  undefined
  model?:     undefined
}

export type RuntimeErrorEnvelope = MaxIterationsErrorEnvelope | AbandonedRunErrorEnvelope
export type AgentErrorEnvelope = ModelErrorEnvelope | RuntimeErrorEnvelope

export interface ToolSchema {
  name:        string
  description: string
  inputSchema: JSONSchema
}

export interface ModelRequest {
  model:           string
  system?:         string          // system prompt (adapter converts to provider format)
  messages:        Message[]
  tools?:          ToolSchema[]
  toolChoice?:     unknown
  responseFormat?: unknown
  reasoning?:      ReasoningOptions
  metadata?:       Record<string, unknown>
  /** #126: sampling temperature. When set, adapters forward it to the provider; when omitted, the param is not sent (provider default). */
  temperature?:    number
  /** Soft cap on completion tokens. OpenAI-compatible adapters default this when omitted. */
  maxTokens?:      number
  /** PR-D Phase 1: when 'system-end', adapter wraps system block with cache_control. */
  cacheBreakpoint?: 'system-end'
}

export interface ModelResponse {
  content:       MessageContent[]
  toolCalls:     ToolCall[]
  usage?:        ModelUsage
  finishReason?: string
  raw?:          unknown
}

export type ModelEvent =
  | { type: 'message_delta'; data: { text: string } }
  | { type: 'tool_call_start'; data: { toolCallId: string; name: string } }
  | { type: 'tool_call_delta'; data: { toolCallId: string; delta: unknown } }
  | { type: 'tool_call_done'; data: { toolCallId: string; input: unknown; invalidArguments?: InvalidToolArguments } }
  | { type: 'usage'; data: ModelUsage }
  | { type: 'error'; data: { code: string; message: string; retryable?: boolean } }

export interface ModelUsage {
  inputTokens:        number
  outputTokens:       number
  cost?:              number
  /** PR-D: tokens served from provider prefix cache (Anthropic). */
  cacheReadTokens?:     number
  /** PR-D: tokens written to provider prefix cache (Anthropic). */
  cacheCreationTokens?: number
}

export interface ReasoningOptions {
  effort?: 'low' | 'medium' | 'high'
  budget?: number
}

export interface IModelGateway {
  complete(request: ModelRequest): Promise<ModelResponse>
  stream(request: ModelRequest): AsyncIterable<ModelEvent>
}
