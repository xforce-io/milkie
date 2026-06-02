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
  | 'wm.mutated'
  | 'tool.emitted'
  | 'agent.checkpoint'
  | 'object.created'
  | 'relation.created'

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
  /**
   * #81: LLM 侧 tool_use id（来自 ModelResponse.toolCalls[].id）。外部消费方
   * （如 alfred turn orchestrator）按此与 tool.responded 配对。旧 trace 无此字段
   * → optional。
   */
  toolCallId?: string
  /** Phase 3: hash of canonicalized (toolName + input); cache key for replay. */
  requestHash: string
}

export interface ToolRespondedPayload {
  toolName: string
  /**
   * #81: 与对应 tool.requested 相同的 LLM 侧 tool_use id，用于配对 requested↔responded。
   * 旧 trace 无此字段 → optional。
   */
  toolCallId?: string
  /**
   * #81: 显式成功/失败标志。'ok' = output 分支，'error' = error 分支。
   * 外部消费方据此判断而无需推导 output/error 的存在性。旧 trace 无此字段 → optional。
   */
  status?: 'ok' | 'error'
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

/**
 * #60: a business FSM event a tool handler emitted via ctx.emit, captured so
 * replay can reproduce the emit-driven transition without re-running the
 * handler (ReplayingIOPort serves cached tool output and never runs the thunk).
 * Recorded once per tool call that WON the FSM's single pendingEvent slot
 * (first-emit-wins); keyed by (toolCallId, occurrence). toolCallId comes from
 * the cached LLM output so it is stable across replay, but provider ids are only
 * unique within ONE response — `occurrence` disambiguates a reused id across the run.
 */
export interface ToolEmittedPayload {
  toolCallId: string
  /** 0-based count of how many times toolCallId was seen in this run before this call. */
  occurrence: number
  event: {
    name:     string
    payload?: unknown
    guard?:   GuardEvaluation[]
  }
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

// ---- Lineage payloads (#37 / #38; vocabulary in docs/lineage-taxonomy.md) ----

/** Core object types — the framework's controlled vocabulary (#39). */
export type CoreObjectType = 'passage' | 'file' | 'claim' | 'artifact-blob'
/**
 * Object type (#113 P4 — extensible). Apps MAY add their own kinds using a
 * `namespace:kind` convention, e.g. `code:function`, `db:row`, so cross-run
 * queries can still group by core kind while distinguishing app kinds (see
 * docs/lineage-taxonomy.md). The `(string & {})` keeps autocomplete for the core
 * set while permitting namespaced extensions — no longer a hard closed union.
 */
export type ObjectType = CoreObjectType | (string & {})

/** Core relation types — the framework's controlled vocabulary (#39). */
export type CoreRelationType = 'cites' | 'derives_from' | 'supersedes' | 'equivalent_to'
/** Relation type (#113 P4 — extensible, same `namespace:kind` convention). */
export type RelationType = CoreRelationType | (string & {})

/**
 * #37: a content-addressable artifact an agent read or produced. Mints a stable
 * `objectId` handle (content-addressed → identical across record/replay). Recorded
 * by RecordingIOPort right after the producing `tool.responded`; replay does not
 * re-run the handler, so this is NOT re-emitted on replay (the cached tool output
 * already carries the id). Pure lineage metadata — does not touch the cache/replay
 * sequence.
 */
export interface ObjectCreatedPayload {
  objectId: string
  type:     ObjectType
  /** The real event that produced the content (e.g. the `tool.responded`). */
  producerEventId: string
  /** Aligns with the producer's `outputHash` (#25) when alignable. */
  hash?:    string
  /** Type-specific locator, e.g. {file, lineStart, lineEnd} for a passage. */
  meta?:    Record<string, unknown>
}

/**
 * #38: a typed, directed edge between two Objects (by objectId). Producer declares
 * it explicitly (e.g. a `cite` tool); the runtime never infers it from LLM text.
 */
export interface RelationCreatedPayload {
  relationId: string
  type:       RelationType
  fromObjectId: string
  toObjectId:   string
  /** The event that triggered the link (onto the causal chain). */
  causedByEventId: string
  meta?:      Record<string, unknown>
}

// ---- Lineage buffer (declared during a tool call, flushed by RecordingIOPort) ----

/** A createObject declaration, buffered during a tool call. */
export interface PendingObject {
  objectId: string
  type:     ObjectType
  hash?:    string
  meta?:    Record<string, unknown>
}

/** A createRelation declaration, buffered during a tool call (#38). */
export interface PendingRelation {
  relationId:   string
  type:         RelationType
  fromObjectId: string
  toObjectId:   string
  meta?:        Record<string, unknown>
}

/**
 * Per-tool-call sink for lineage declarations. `ctx.createObject` /
 * `ctx.createRelation` push here during the handler; RecordingIOPort drains it
 * right after appending `tool.responded` (so `producerEventId` / `causedByEventId`
 * resolve to that event). On replay the handler never runs → buffer stays empty.
 */
export interface LineageBuffer {
  objects:   PendingObject[]
  relations: PendingRelation[]
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
export type ToolEmittedEvent       = Event<ToolEmittedPayload>       & { type: 'tool.emitted' }
export type SkillLoadedEvent       = Event<SkillLifecyclePayload>    & { type: 'skill.loaded' }
export type SkillUnloadedEvent     = Event<SkillLifecyclePayload>    & { type: 'skill.unloaded' }
export type ObjectCreatedEvent     = Event<ObjectCreatedPayload>     & { type: 'object.created' }
export type RelationCreatedEvent   = Event<RelationCreatedPayload>   & { type: 'relation.created' }

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
  | ToolEmittedEvent
  | SkillLoadedEvent
  | SkillUnloadedEvent
  | ObjectCreatedEvent
  | RelationCreatedEvent
