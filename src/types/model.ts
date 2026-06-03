import type { Message, MessageContent } from './common.js'
import type { ToolCall } from './tool.js'
import type { JSONSchema } from './common.js'

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
  | { type: 'tool_call_done'; data: { toolCallId: string; input: unknown } }
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
