// Core framework entry point

export { Milkie } from './runtime/Milkie.js'
export { AgentRuntime } from './runtime/AgentRuntime.js'
export { AgentFactory } from './runtime/AgentFactory.js'
export {
  DefaultIOPort,
  resolveIOInvocationControl,
} from './runtime/IOPort.js'
export type {
  IIOPort,
  LLMInvocationOptions,
  ToolInvocationOptions,
} from './runtime/IOPort.js'
export { FSMEngine } from './fsm/FSMEngine.js'
export { ContextRegions } from './context/ContextRegions.js'
export { assemble, type AssembleScope, type AssembledContext } from './context/assemble.js'
export { ToolRegistry } from './tools/ToolRegistry.js'
export { WorkingMemory } from './store/WorkingMemory.js'

// State Stores
export { MemoryStore } from './store/MemoryStore.js'
export { SQLiteStore } from './store/SQLiteStore.js'
export { RedisStore } from './store/RedisStore.js'

// Trajectory Recorders
export { NoopRecorder } from './trajectory/NoopRecorder.js'
export { InMemoryRecorder } from './trajectory/InMemoryRecorder.js'
export { JSONLRecorder } from './trajectory/JSONLRecorder.js'
export { ConsoleRecorder } from './trajectory/ConsoleRecorder.js'
export { TrajectoryStore } from './trajectory/TrajectoryStore.js'

// Model Gateway
export { AnthropicAdapter }         from './gateway/AnthropicAdapter.js'
export { OpenAICompatibleAdapter }  from './gateway/OpenAICompatibleAdapter.js'
export { ModelGatewayError, normalizeModelGatewayError } from './gateway/ModelGatewayError.js'
export { createGateway }            from './gateway/GatewayFactory.js'

// #84: portable session export/import
export { PORTABLE_SESSION_SCHEMA_VERSION } from './runtime/PortableSession.js'
export type { PortableSession } from './runtime/PortableSession.js'

// Trace stores / views
export { MemoryEventStore } from './trace/MemoryEventStore.js'
export { JsonlEventStore } from './trace/JsonlEventStore.js'
export { MemoryTraceObjectStore, FileTraceObjectStore } from './trace/TraceObjectStore.js'
export { contextAt, contextBefore, getRegionAt, getRegionBefore } from './trace/RegionContextView.js'
export type { ITraceObjectStore, ICrashSafeTraceObjectStore } from './trace/TraceObjectStore.js'
export { isCrashSafeTraceObjectStore } from './trace/TraceObjectStore.js'
export type { IEventStore, ICrashSafeEventStore } from './trace/EventStore.js'
export { isCrashSafeEventStore } from './trace/EventStore.js'
export type { RegionContentRef, ContextFoldMode } from './trace/RegionContextView.js'

// #227 / s-017: immutable task outcome finalization
export {
  MemoryTaskOutcomeFinalizationStore,
  FileTaskOutcomeFinalizationStore,
} from './outcome/TaskOutcomeFinalizationStore.js'
export type {
  ITaskOutcomeFinalizationStore,
  FileFinalizationFsOps,
  FileTaskOutcomeFinalizationStoreOptions,
} from './outcome/TaskOutcomeFinalizationStore.js'

// Built-in tools
export { cognitiveTools } from './tools/cognitive.js'
export { systemTools } from './tools/system.js'
export { BUILTIN_TOOL_NAMES, resolveEffectiveBuiltinTools } from './tools/builtinTools.js'
export { RunControl, RunControlError } from './runtime/RunControl.js'

// Types
export type {
  AgentConfig,
  BuiltinToolName,
  BuiltinToolPolicy,
  FSMDefinition,
  FSMState,
  ModelConfig,
} from './types/agent.js'

export { summarizeRun, parseJsonlEvents, TraceInspectError } from './trace/summarizeRun.js'
export type { RunSummary } from './trace/summarizeRun.js'

export type {
  AgentInvokeRequest,
  AgentResult,
  ArtifactRef,
  AttachProjectionRequest,
  BudgetFinalizeContext,
  ContextProjection,
  DeliverableSpec,
  ProjectionBound,
  StopReason,
  TaskResult,
  Message,
  MessageContent,
  ImageMediaType,
  ImageSource,
  RunControlOptions,
  JSONSchema,
} from './types/common.js'

export type {
  TaskOutcome,
  TaskOutcomeValue,
  TaskOutcomeSource,
  TaskOutcomeScore,
  RecordTaskOutcomeInput,
  VerifierClaimType,
  VerifierClaim,
  EvidenceRef,
  FinalizeTaskOutcomeInput,
  TaskOutcomeFinalization,
  FinalizationConflictKind,
  FinalizationAttemptResult,
  DurabilityClass,
  TaskOutcomeEvidenceErrorReason,
  TaskOutcomeFinalizationStoreErrorKind,
} from './types/outcome.js'
export {
  TaskOutcomeError,
  TaskOutcomeRunNotFoundError,
  TaskOutcomeFinalizationValidationError,
  TaskOutcomeFinalizationConfigurationError,
  TaskOutcomeEvidenceError,
  TaskOutcomeFinalizationStoreError,
  TaskOutcomeFinalizationCorruptionError,
} from './types/outcome.js'

export type {
  ToolDefinition,
  ToolContext,
  ToolCall,
  ToolResult,
} from './types/tool.js'

export {
  IOControlError,
  IOInvocationValidationError,
  LlmInvocationError,
} from './types/model.js'
export type {
  IModelGateway,
  ModelRequest,
  ModelResponse,
  ModelEvent,
  ModelErrorCode,
  ModelErrorEnvelope,
  ModelErrorPhase,
  ModelCapabilities,
  ModelGatewayCallOptions,
  AgentErrorEnvelope,
  GatewayInvocationOptions,
  IOControlErrorCode,
  IOControlErrorEnvelope,
  IOControlOperation,
  IOInvocationControl,
  LlmInvocationFailureEnvelope,
  ToolSchema,
  RunDeadlineExceededErrorEnvelope,
  RunCancelledErrorEnvelope,
} from './types/model.js'

// #229: LLM failure replay public errors
export { TraceWriteError } from './trace/TraceWriteError.js'
export type { TraceWriteErrorDetails, TraceWriteStage } from './trace/TraceWriteError.js'
export { TraceIntegrityError } from './trace/TraceIntegrityError.js'
export type { TraceIntegrityErrorDetails, TraceIntegrityErrorKind } from './trace/TraceIntegrityError.js'
export type {
  LlmOutcome,
  LlmSuccessOutcome,
  LlmFailureOutcome,
  LlmFailureView,
} from './trace/LlmOutcome.js'
export { decodeLlmOutcome, reconstructLlmError } from './trace/LlmOutcome.js'
export type { RecordedLlmFailureEnvelope, TrustedProviderFamily } from './trace/types.js'
export { LLM_OUTCOME_SCHEMA_VERSION } from './trace/types.js'

export type {
  IStateStore,
  AgentCheckpoint,
  AgentEvent,
} from './types/store.js'

export type {
  ITrajectoryRecorder,
  Trajectory,
  Span,
  SpanAttributes,
} from './types/trajectory.js'
