import type { ModelRequest, ModelResponse } from '../types/model.js'

/**
 * Agent Trace event types.
 *
 * Phase 3 adds lifecycle events and content-addressed cache fields on I/O
 * payloads. Agent spawn and return events are now included; fork events remain future.
 */
export type EventKind =
  | 'llm.requested'
  | 'llm.responded'
  | 'tool.requested'
  | 'tool.responded'
  | 'agent.run.started'
  | 'agent.run.completed'
  | 'clock.read'
  | 'uuid.generated'
  | 'region.added'
  | 'region.removed'
  | 'context.boundary.applied'
  | 'fsm.transition'
  | 'skill.loaded'
  | 'skill.unloaded'
  | 'agent.spawned'
  | 'agent.returned'

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
  /** PR-D: cache-health snapshot lifted from response.usage; null when provider does not report. */
  cacheStats?: {
    readTokens:       number
    creationTokens:   number
    totalInputTokens: number
    /** readTokens / totalInputTokens, [0, 1]. 0 when totalInputTokens === 0. */
    hitRate:          number
  }
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
  /** #25: 成功时 hashCanonical(output) → "sha256:..."；error 分支不填。 */
  outputHash?:   string
  /** #25: canonicalize(output) 的 UTF-8 字节数（= 对象库中那份大小）。 */
  outputBytes?:  number
  /** #25: 为 #37 (object.created) 预留；本期不填。 */
  artifactRefs?: string[]
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

export interface AgentSpawnedPayload {
  /** 父 AgentRuntime.agentRunId。 */
  parentRunId: string
  /**
   * 子运行的独立 runId（父在 spawn 时铸造）。子的 I/O 事件以此 id 存入 event store，
   * 可经 Milkie.replay(childRunId) 独立重放。
   */
  childRunId:  string
  agentId:     string
  goal:        string
}

export interface AgentReturnedPayload {
  childRunId: string
  status:     'completed' | 'interrupted' | 'error'
}

// ---- Non-determinism payloads (Phase 4) ----

export interface ClockReadPayload {
  /** Epoch ms returned by the underlying clock at the time agent code called port.now(). */
  value: number
}

export interface UuidGeneratedPayload {
  /** UUID string returned by the underlying generator at the time agent code called port.uuid(). */
  value: string
}

// ---- Region / context boundary payloads (Phase 4.6) ----

export interface RegionAddedPayload {
  /** Region id (e.g. 'header', 'skill:verifier', 'scratch:abc123'). */
  id:        string
  /** Stable identifier for which substrate section/target this region targets. */
  target:    'system' | 'message' | 'tool'
  section:   string
  stability: 'immutable' | 'session-stable' | 'turn-stable' | 'volatile'
  /** Why this region appeared (e.g. 'agent-set', 'turn-archived', 'promoted-to-wm'). */
  reason:    string
  /** Content-addressed canonical Region.content bytes. Missing on legacy events or unsupported content shapes. */
  contentHash?: string
  /** Content-addressed canonical format(content) bytes, when renderable. */
  renderedHash?: string
}

export interface RegionRemovedPayload {
  id:     string
  /** Why this region was removed (e.g. 'turn-local-released', 'ttl-expired', 'promoted-source-removed'). */
  reason: string
}

export interface ContextBoundaryAppliedPayload {
  /** Which boundary engine fired: 'turn-end' (crystallization) for now. */
  boundary: 'turn-end' | 'turn-start' | 'fsm-step'
  /** epoch of the regions Map AFTER the boundary engine ran. */
  epoch:    number
  /** Summary of crystallization activity (omitted when boundary is non-turn-end). */
  crystallization?: {
    kept:         number   // count, not full ids (full ids in region.added/removed events)
    dropped:      number
    promoted:     number
    archivedPair: string | undefined   // id of the new history pair region, if any
  }
}

// ---- Skill lifecycle payloads (#22) ----

export interface SkillLifecyclePayload {
  skillId: string
  version: string
  source:  string
  sha?:    string
}

// ---- FSM payloads (#21) ----

/**
 * FSM event domain — explicit taxonomy of the previously-conflated FSMEvent.name
 * namespace. Captured at emit site, not inferred post-hoc.
 *
 *   lifecycle       — FSM/Runtime phase markers (DONE, DIRECT, RESUME)
 *   signal          — global override events that force a transition regardless
 *                     of current state's `on:` map (interrupt, error global path)
 *   runtime-control — AgentRuntime-driven control flow (retry, error escalation)
 *   business        — tool handlers via ctx.emit(); user-defined names that
 *                     participate in the state machine's `on:` table
 */
export type FsmEventDomain = 'lifecycle' | 'signal' | 'runtime-control' | 'business'

export interface GuardEvaluation {
  /** 判断标识,如 'intent-threshold'。 */
  guardId:      string
  /** 判断结果:产出的事件名或布尔/任意值。 */
  result:       unknown
  /** 决定结果真假的最小输入切片(约定最小化,框架不强制)。 */
  contextSlice: unknown
}

export interface FsmTransitionPayload {
  from: string
  to:   string
  trigger: {
    domain: FsmEventDomain
    name:   string
    /** Optional structured payload (omit for lifecycle/signal which carry none). */
    payload?: unknown
  }
  /** #31:本次转移背后的判断依据(工具自报,可选)。 */
  guardEvaluations?: GuardEvaluation[]
}

// ---- Typed event aliases ----

export type LlmRequestedEvent       = Event<LlmRequestedPayload>       & { type: 'llm.requested' }
export type LlmRespondedEvent       = Event<LlmRespondedPayload>       & { type: 'llm.responded' }
export type ToolRequestedEvent      = Event<ToolRequestedPayload>      & { type: 'tool.requested' }
export type ToolRespondedEvent      = Event<ToolRespondedPayload>      & { type: 'tool.responded' }
export type AgentRunStartedEvent    = Event<AgentRunStartedPayload>    & { type: 'agent.run.started' }
export type AgentRunCompletedEvent  = Event<AgentRunCompletedPayload>  & { type: 'agent.run.completed' }
export type AgentSpawnedEvent       = Event<AgentSpawnedPayload>       & { type: 'agent.spawned' }
export type AgentReturnedEvent      = Event<AgentReturnedPayload>      & { type: 'agent.returned' }
export type ClockReadEvent          = Event<ClockReadPayload>          & { type: 'clock.read' }
export type UuidGeneratedEvent      = Event<UuidGeneratedPayload>      & { type: 'uuid.generated' }
export type RegionAddedEvent        = Event<RegionAddedPayload>        & { type: 'region.added' }
export type RegionRemovedEvent      = Event<RegionRemovedPayload>      & { type: 'region.removed' }
export type ContextBoundaryAppliedEvent = Event<ContextBoundaryAppliedPayload> & { type: 'context.boundary.applied' }
export type FsmTransitionEvent     = Event<FsmTransitionPayload>     & { type: 'fsm.transition' }
export type SkillLoadedEvent       = Event<SkillLifecyclePayload>    & { type: 'skill.loaded' }
export type SkillUnloadedEvent     = Event<SkillLifecyclePayload>    & { type: 'skill.unloaded' }

export type AnyEvent =
  | LlmRequestedEvent
  | LlmRespondedEvent
  | ToolRequestedEvent
  | ToolRespondedEvent
  | AgentRunStartedEvent
  | AgentRunCompletedEvent
  | AgentSpawnedEvent
  | AgentReturnedEvent
  | ClockReadEvent
  | UuidGeneratedEvent
  | RegionAddedEvent
  | RegionRemovedEvent
  | ContextBoundaryAppliedEvent
  | FsmTransitionEvent
  | SkillLoadedEvent
  | SkillUnloadedEvent
