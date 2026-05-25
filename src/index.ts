// Core framework entry point

export { Milkie } from './runtime/Milkie.js'
export { AgentRuntime } from './runtime/AgentRuntime.js'
export { AgentFactory } from './runtime/AgentFactory.js'
export { FSMEngine } from './fsm/FSMEngine.js'
export { ContextRegions } from './context/ContextRegions.js'
export { assemble, type AssembleScope, type AssembledContext } from './context/assemble.js'
export { ToolRegistry } from './tools/ToolRegistry.js'
export { WorkingMemory } from './store/WorkingMemory.js'
export { CheckpointManager } from './store/CheckpointManager.js'

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
export { createGateway }            from './gateway/GatewayFactory.js'

// Built-in tools
export { cognitiveTools } from './tools/cognitive.js'
export { systemTools } from './tools/system.js'

// Types
export type {
  AgentConfig,
  FSMDefinition,
  FSMState,
  ModelConfig,
} from './types/agent.js'

export type {
  AgentInvokeRequest,
  AgentResult,
  TaskResult,
  Message,
  MessageContent,
  JSONSchema,
} from './types/common.js'

export type {
  ToolDefinition,
  ToolContext,
  ToolCall,
  ToolResult,
} from './types/tool.js'

export type {
  IModelGateway,
  ModelRequest,
  ModelResponse,
  ModelEvent,
  ToolSchema,
} from './types/model.js'

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
