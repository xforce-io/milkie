import { v4 as uuidv4 } from 'uuid'
import type { ModelRequest, ModelResponse, ModelEvent, IModelGateway } from '../types/model.js'
import { aggregateStream } from '../gateway/StreamAggregator.js'

/**
 * IOPort — the Agent Runtime's declared boundary for non-deterministic
 * effects.
 *
 * All non-deterministic operations — LLM calls, tool invocations, clock reads,
 * UUID generation — pass through IOPort. This is the target architecture's
 * decoration point: future Agent Trace (event log + content-addressed cache +
 * non-determinism log) plugs in here to record events and serve cached
 * responses during replay, without the rest of the runtime knowing.
 *
 * Per ARCHITECTURE.md cross-cutting invariants:
 *   - "IOPort is part of Agent Runtime's design, not an Agent Trace-imposed
 *      hook." (#2)
 *   - "Agent Runtime does not depend on Agent Trace. Runtime depends only on
 *      its IOPort contract." (#3)
 *
 * Implementations:
 *   - DefaultIOPort: passthrough to gateway / Date.now() / uuid() — current
 *     production behavior.
 *   - (target) RecordingIOPort: writes llm.requested / llm.responded events
 *     plus non-determinism events to an Agent Trace event log.
 *   - (target) ReplayIOPort: serves cached responses by request hash, returns
 *     recorded clock / UUID values from the non-determinism log.
 */
export interface IIOPort {
  /**
   * Invoke a language model.
   *
   * When `onEvent` is provided, the call streams: each `ModelEvent` is
   * forwarded to `onEvent` as it arrives and the fragments are aggregated into
   * the resolved `ModelResponse`. When omitted, a single non-streaming
   * completion is performed.
   */
  invokeLLM(
    request: ModelRequest,
    onEvent?: (e: ModelEvent) => void,
  ): Promise<ModelResponse>

  /**
   * Invoke a tool. The `execute` thunk is what actually runs the tool's
   * handler; an IOPort may choose to call it (default), record around it
   * (recording), or replace it with a cached result (replay).
   *
   * `toolName` and `input` are exposed so future implementations can hash
   * them as a cache key without inspecting the thunk.
   */
  invokeTool(
    toolName: string,
    input: unknown,
    execute: () => Promise<unknown>,
  ): Promise<unknown>

  /** Current epoch milliseconds. Replacement for direct `Date.now()`. */
  now(): number

  /** A new UUID. Replacement for direct `uuid()`. */
  uuid(): string
}

/**
 * Default IOPort: direct passthrough. No recording, no caching, no replay.
 * This is what production runs use today; Agent Trace decoration is added
 * by wrapping or replacing this implementation.
 */
export class DefaultIOPort implements IIOPort {
  constructor(private readonly gateway: IModelGateway) {}

  invokeLLM(
    request: ModelRequest,
    onEvent?: (e: ModelEvent) => void,
  ): Promise<ModelResponse> {
    if (onEvent) {
      return aggregateStream(this.gateway.stream(request), onEvent)
    }
    return this.gateway.complete(request)
  }

  invokeTool(
    _toolName: string,
    _input: unknown,
    execute: () => Promise<unknown>,
  ): Promise<unknown> {
    return execute()
  }

  now(): number {
    return Date.now()
  }

  uuid(): string {
    return uuidv4()
  }
}
