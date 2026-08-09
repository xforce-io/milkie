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

export interface IOInvocationControl {
  readonly signal?: AbortSignal
  readonly deadlineAt?: number
}

export interface GatewayInvocationOptions {
  readonly signal?: AbortSignal
}

export type IOControlOperation = 'llm' | 'tool'
export type IOControlErrorCode = 'IO_CANCELLED' | 'IO_DEADLINE_EXCEEDED'

export interface IOControlErrorEnvelope {
  code:      IOControlErrorCode
  message:   'I/O invocation was cancelled.' | 'I/O invocation deadline exceeded.'
  phase:     'io_control'
  operation: IOControlOperation
  retryable: false
  provider?: undefined
  model?:    undefined
}

export class IOInvocationValidationError extends Error {
  readonly code = 'IO_INVALID_DEADLINE' as const
  readonly retryable = false as const

  constructor() {
    super('I/O invocation deadline must be a finite non-negative Unix epoch millisecond.')
    this.name = 'IOInvocationValidationError'
  }
}

export class IOControlError extends Error {
  readonly code: IOControlErrorCode
  readonly phase = 'io_control' as const
  readonly operation: IOControlOperation
  readonly retryable = false as const
  readonly envelope: IOControlErrorEnvelope

  constructor(code: IOControlErrorCode, operation: IOControlOperation) {
    const message = code === 'IO_CANCELLED'
      ? 'I/O invocation was cancelled.' as const
      : 'I/O invocation deadline exceeded.' as const
    super(message)
    this.name = 'IOControlError'
    this.code = code
    this.operation = operation
    this.envelope = {
      code,
      message,
      phase: 'io_control',
      operation,
      retryable: false,
    }
  }
}

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
export type AgentErrorEnvelope = ModelErrorEnvelope | RuntimeErrorEnvelope | IOControlErrorEnvelope

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
  complete(request: ModelRequest, options?: GatewayInvocationOptions): Promise<ModelResponse>
  stream(request: ModelRequest, options?: GatewayInvocationOptions): AsyncIterable<ModelEvent>
}
