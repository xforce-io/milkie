import type { AgentErrorEnvelope, IOInvocationControl, ModelEvent } from './model.js'

export type JSONValue = string | number | boolean | null | JSONValue[] | { [k: string]: JSONValue }
export type JSONObject = Record<string, JSONValue>
export type JSONSchema = Record<string, unknown>

export interface InvalidToolArguments {
  code:      'TOOL_ARGUMENTS_INVALID_JSON' | 'TOOL_ARGUMENTS_TRUNCATED'
  message:   string
  rawLength?: number
}


export interface Message {
  role: 'user' | 'assistant' | 'tool'
  content: MessageContent[]
}

export type MessageContent =
  | { type: 'text'; text: string }
  | { type: 'tool_use'; id: string; name: string; input: unknown; invalidArguments?: InvalidToolArguments }
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
  /** Absolute deadline and caller cancellation signal for this invocation. */
  control?: IOInvocationControl
  /**
   * #247: this-run deliverable contract. Key present (including `[]`) replaces
   * the agent default wholesale. Key omitted → agent default, or no contract.
   */
  deliverables?: DeliverableSpec[]
}

/** #244: why the run loop stopped. Independent of task outcome. */
export type StopReason =
  | 'model_stop'
  | 'budget_exhausted'
  | 'deadline'
  | 'cancelled'
  | 'interrupted'
  | 'runtime_error'

export type ArtifactType = 'file' | 'object'

/** #247: declared target deliverable. */
export interface DeliverableSpec {
  name: string
  type: ArtifactType
  path?: string
  required?: boolean
}

/** #247 / #244: one item in the returned artifacts list. */
export interface ArtifactRef {
  name: string
  type: ArtifactType
  path?: string
  objectId?: string
  state: 'produced' | 'missing'
  hash?: string
}

/** #244: optional finalize hook after budget/deadline stop. */
export interface BudgetFinalizeContext {
  recordArtifact: (artifact: Omit<ArtifactRef, 'state'> & { state?: 'produced' }) => void
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
  /** #244: why the loop stopped. Never `goal_completed`. */
  stopReason:  StopReason
  /** Diagnostic code (MAX_ITERATIONS_EXCEEDED, RUN_DEADLINE_EXCEEDED, IO_*, …). */
  stopCode?:   string
  /** #244/#247: incomplete delivery or non-natural stop. */
  partial:     boolean
  checkpointId?: string
  /** #247: declared/produced artifacts. Never a directory listing. */
  artifacts:   ArtifactRef[]
  error?:      AgentErrorEnvelope
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
