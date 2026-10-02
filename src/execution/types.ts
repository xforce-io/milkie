import type { ConnectionInput, ConnectionProjection } from '../connection/types.js'

export type ExecutionStatus = 'starting' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'timed_out' | 'unknown'
export type ExecutionCode = 'invalid_request' | 'unsupported_runtime' | 'unsupported_constraint' | 'context_not_found' | 'connection_mismatch' | 'context_busy' | 'config_missing' | 'session_missing' | 'session_mismatch' | 'auth_failed' | 'process_failed' | 'native_cancelled' | 'protocol_error' | 'storage_error' | 'platform_unsupported' | 'policy_mismatch'
export class ExecutionError extends Error {
  constructor(readonly code: ExecutionCode) {
    super(`Execution request failed: ${code}.`)
    this.name = 'ExecutionError'
  }
}
/** JSON Schema subset accepted for a host tool. Omitted additionalProperties means false. */
export interface HostToolSchema {
  type: 'object' | 'string' | 'number' | 'integer' | 'boolean' | 'array'
  properties?: Record<string, HostToolSchema>
  required?: string[]
  additionalProperties?: boolean
  items?: HostToolSchema
}
export interface HostToolSpec {
  name: string
  description: string
  inputSchema: HostToolSchema
}
export interface ToolCall {
  callId: string
  /** Present only when this runtime's capability says the CLI provides one. */
  nativeCallId?: string
  name: string
  input: unknown
  runId: string
  contextId: string
}
export type ToolResult = { ok: true; output: string } | { ok: false; code: 'invalid_input' | 'rejected'; message: string }
export type ToolHandler = (call: ToolCall) => Promise<ToolResult> | ToolResult
export interface ToolCallRecord {
  version: 1
  callId: string
  nativeCallId?: string
  name: string
  input: unknown
  runId: string
  contextId: string
  status: 'pending' | 'succeeded' | 'invalid_input' | 'rejected' | 'reconciled'
  output?: string
  message?: string
}
export interface ExecutionConstraints {
  toolPolicy?: 'read-only' | 'standard'
  timeoutMs?: number
  /** Host-executed tools. Mutually exclusive with toolPolicy. */
  tools?: HostToolSpec[]
  /** Default serial. Only valid together with tools. */
  forwarding?: 'serial' | 'parallel'
}
/** Host-prepared CLI config/login directory and session directory. */
export interface CliStorage {
  configDir: string
  sessionDir: string
}
export interface ExecutionContext {
  version: 1
  contextId: string
  connection: ConnectionProjection
  cwd: string
  /** Absolute real paths. Absent only for API transport contexts. */
  configDir?: string
  sessionDir?: string
  nativeSessionId?: string
  nativeSessionFile?: string
  hasExecuted: boolean
}
export interface ExecutionRecord {
  version: 1
  runId: string
  contextId: string
  nativeSessionId?: string
  status: ExecutionStatus
  code?: ExecutionCode
  startedAt: number
  finishedAt?: number
  heartbeatAt: number
  stopped: boolean
  /** Process identities only; never argv or environment. */
  resources?: Array<{ pid: number; startedAt: string }>
  output?: string
}
export interface ExecutionCapabilities {
  /** Adapter support is distinct from installed/authenticated readiness. */
  availability: 'unchecked'
  supported: boolean
  code?: 'unsupported_runtime' | 'platform_unsupported'
  resume: boolean
  workingDirectory: boolean
  toolPolicies: Array<'read-only' | 'standard'>
  /** CLI runtimes can register host tools. API transport cannot. */
  hostTools: boolean
  /** True when a host-tool call includes the CLI's own tool-call id. */
  nativeCallId: boolean
  forwarding: Array<'serial' | 'parallel'>
  timeout: boolean
  cancel: boolean
}
export interface ExecutionClientOptions {
  connection: ConnectionInput
  dataDir: string
  /** Explicit child environment; never persisted or included in diagnostics. */
  env?: NodeJS.ProcessEnv
}
export interface WorkerRequest {
  dataDir: string
  context: ExecutionContext
  record: ExecutionRecord
  connection: ConnectionInput
  input: string
  constraints: { toolPolicy?: 'read-only' | 'standard'; timeoutMs: number }
  hostTools?: { tools: HostToolSpec[]; forwarding: 'serial' | 'parallel' }
}
export interface WorkerToolMessage { type: 'tool-call'; call: ToolCall }
export interface WorkerToolReply { type: 'tool-result'; callId: string; result: unknown }
