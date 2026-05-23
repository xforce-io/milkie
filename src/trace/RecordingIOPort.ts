import type { ModelRequest, ModelResponse } from '../types/model.js'
import type { IIOPort } from '../runtime/IOPort.js'
import type { IEventStore } from './EventStore.js'
import type {
  LlmRequestedPayload,
  LlmRespondedPayload,
  ToolRequestedPayload,
  ToolRespondedPayload,
} from './types.js'

/**
 * RecordingIOPort — decorates any inner IOPort to emit paired
 * `requested` / `responded` events for every LLM and tool call.
 *
 * Phase 2 scope: records LLM and tool I/O only. Clock (`now()`) and
 * UUID generation (`uuid()`) pass through to the inner port — they
 * will be recorded in Phase 4 (non-determinism log).
 *
 * The inner port can be any IIOPort implementation:
 *   - DefaultIOPort: live calls
 *   - (future) replay impls: cached responses
 *   - test mocks
 *
 * Per ARCHITECTURE.md cross-cutting invariants:
 *   "IOPort is part of Agent Runtime's design, not an Agent
 *    Trace-imposed hook."
 *
 * RecordingIOPort is the concrete realization of Agent Trace's
 * "decorator implementation of that port".
 */
export class RecordingIOPort implements IIOPort {
  constructor(
    private readonly inner: IIOPort,
    private readonly store: IEventStore,
    private readonly runId: string,
    private readonly actor: string = 'runtime',
  ) {}

  async invokeLLM(request: ModelRequest): Promise<ModelResponse> {
    const reqEventId = this.inner.uuid()
    await this.store.append({
      id:        reqEventId,
      runId:     this.runId,
      type:      'llm.requested',
      actor:     this.actor,
      timestamp: this.inner.now(),
      payload:   { request } satisfies LlmRequestedPayload,
    })

    const response = await this.inner.invokeLLM(request)

    await this.store.append({
      id:        this.inner.uuid(),
      runId:     this.runId,
      type:      'llm.responded',
      actor:     this.actor,
      causedBy:  reqEventId,
      timestamp: this.inner.now(),
      payload:   { response } satisfies LlmRespondedPayload,
    })

    return response
  }

  async invokeTool(
    toolName: string,
    input: unknown,
    execute: () => Promise<unknown>,
  ): Promise<unknown> {
    const reqEventId = this.inner.uuid()
    await this.store.append({
      id:        reqEventId,
      runId:     this.runId,
      type:      'tool.requested',
      actor:     this.actor,
      timestamp: this.inner.now(),
      payload:   { toolName, input } satisfies ToolRequestedPayload,
    })

    try {
      const output = await this.inner.invokeTool(toolName, input, execute)
      await this.store.append({
        id:        this.inner.uuid(),
        runId:     this.runId,
        type:      'tool.responded',
        actor:     this.actor,
        causedBy:  reqEventId,
        timestamp: this.inner.now(),
        payload:   { toolName, output } satisfies ToolRespondedPayload,
      })
      return output
    } catch (err) {
      await this.store.append({
        id:        this.inner.uuid(),
        runId:     this.runId,
        type:      'tool.responded',
        actor:     this.actor,
        causedBy:  reqEventId,
        timestamp: this.inner.now(),
        payload:   {
          toolName,
          error: err instanceof Error ? err.message : String(err),
        } satisfies ToolRespondedPayload,
      })
      throw err
    }
  }

  now(): number {
    return this.inner.now()
  }

  uuid(): string {
    return this.inner.uuid()
  }
}
