import type { ConnectionInput, ConnectionProjection } from '../connection/types.js'

export type ExecutionStatus = 'starting' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'timed_out' | 'unknown'
export type ExecutionCode = 'invalid_request' | 'unsupported_runtime' | 'unsupported_constraint' | 'context_not_found' | 'connection_mismatch' | 'context_busy' | 'config_missing' | 'session_missing' | 'session_mismatch' | 'auth_failed' | 'process_failed' | 'native_cancelled' | 'protocol_error' | 'storage_error' | 'platform_unsupported'
export class ExecutionError extends Error {
  constructor(readonly code: ExecutionCode) {
    super(`Execution request failed: ${code}.`)
    this.name = 'ExecutionError'
  }
}
export interface ExecutionConstraints {
  toolPolicy?: 'read-only' | 'standard'
  timeoutMs?: number
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
  constraints: Required<ExecutionConstraints>
}
