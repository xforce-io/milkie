import type { ModelRequest, ModelResponse } from '../types/model.js'
import type { IIOPort } from '../runtime/IOPort.js'
import type { IEventStore } from './EventStore.js'
import type {
  LlmRequestedPayload,
  LlmRespondedPayload,
  ToolRequestedPayload,
  ToolRespondedPayload,
  AgentRunStartedPayload,
  AgentRunCompletedPayload,
} from './types.js'
import { hashModelRequest, hashToolCall } from './hash.js'

/**
 * RecordingIOPort — decorates an inner IOPort to emit Agent Trace events.
 *
 * Phase 3 additions:
 *  - LLM/tool events carry requestHash (cache key)
 *  - attach()/detach() emit agent.run.started/completed lifecycle events
 *  - Tool errors are recorded as structured payloads (preserve retryable/code/name)
 *
 * Clock (now()) and UUID generation pass through to inner — non-determinism
 * log is Phase 4.
 */
export class RecordingIOPort implements IIOPort {
  constructor(
    private readonly inner: IIOPort,
    private readonly store: IEventStore,
    private readonly runId: string,
    private readonly actor: string = 'runtime',
  ) {}

  async attach(payload: AgentRunStartedPayload): Promise<void> {
    await this.store.append({
      id:        this.inner.uuid(),
      runId:     this.runId,
      type:      'agent.run.started',
      actor:     this.actor,
      timestamp: this.inner.now(),
      payload,
    })
  }

  async detach(payload: AgentRunCompletedPayload): Promise<void> {
    await this.store.append({
      id:        this.inner.uuid(),
      runId:     this.runId,
      type:      'agent.run.completed',
      actor:     this.actor,
      timestamp: this.inner.now(),
      payload,
    })
  }

  async invokeLLM(request: ModelRequest): Promise<ModelResponse> {
    const requestHash = hashModelRequest(request)
    const reqEventId  = this.inner.uuid()
    await this.store.append({
      id:        reqEventId,
      runId:     this.runId,
      type:      'llm.requested',
      actor:     this.actor,
      timestamp: this.inner.now(),
      payload:   { request, requestHash } satisfies LlmRequestedPayload,
    })

    const response = await this.inner.invokeLLM(request)

    await this.store.append({
      id:        this.inner.uuid(),
      runId:     this.runId,
      type:      'llm.responded',
      actor:     this.actor,
      causedBy:  reqEventId,
      timestamp: this.inner.now(),
      payload:   { response, requestHash } satisfies LlmRespondedPayload,
    })

    return response
  }

  async invokeTool(
    toolName: string,
    input: unknown,
    execute: () => Promise<unknown>,
  ): Promise<unknown> {
    const requestHash = hashToolCall(toolName, input)
    const reqEventId  = this.inner.uuid()
    await this.store.append({
      id:        reqEventId,
      runId:     this.runId,
      type:      'tool.requested',
      actor:     this.actor,
      timestamp: this.inner.now(),
      payload:   { toolName, input, requestHash } satisfies ToolRequestedPayload,
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
        payload:   { toolName, output, requestHash } satisfies ToolRespondedPayload,
      })
      return output
    } catch (err) {
      const e = err as { message?: string; retryable?: boolean; code?: string; name?: string }
      const errorPayload: NonNullable<ToolRespondedPayload['error']> = {
        message: e.message ?? String(err),
      }
      if (typeof e.retryable === 'boolean') errorPayload.retryable = e.retryable
      if (typeof e.code === 'string')       errorPayload.code      = e.code
      // 'Error' is the default name; omit it as it carries no information
      if (typeof e.name === 'string' && e.name !== 'Error') errorPayload.name = e.name

      await this.store.append({
        id:        this.inner.uuid(),
        runId:     this.runId,
        type:      'tool.responded',
        actor:     this.actor,
        causedBy:  reqEventId,
        timestamp: this.inner.now(),
        payload:   { toolName, error: errorPayload, requestHash } satisfies ToolRespondedPayload,
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
