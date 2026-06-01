import type { ModelRequest, ModelResponse, ModelEvent } from '../types/model.js'
import type { IIOPort } from '../runtime/IOPort.js'
import type { IEventStore } from './EventStore.js'
import type {
  LlmRequestedPayload,
  LlmRespondedPayload,
  ToolRequestedPayload,
  ToolRespondedPayload,
  AgentRunStartedPayload,
  AgentRunCompletedPayload,
  ClockReadPayload,
  UuidGeneratedPayload,
} from './types.js'
import { hashModelRequest, hashToolCall, canonicalize, contentAddressForCanonicalBytes } from './hash.js'
import type { ITraceObjectStore } from './TraceObjectStore.js'
import type { CausalCursor } from './CausalCursor.js'

function cacheStatsFrom(response: ModelResponse): {
  readTokens:       number
  creationTokens:   number
  totalInputTokens: number
  hitRate:          number
} | undefined {
  const usage = response.usage
  if (!usage || usage.cacheReadTokens === undefined) return undefined
  const readTokens     = usage.cacheReadTokens
  const creationTokens = usage.cacheCreationTokens ?? 0
  const totalInputTokens = usage.inputTokens
  return {
    readTokens,
    creationTokens,
    totalInputTokens,
    hitRate: totalInputTokens > 0 ? readTokens / totalInputTokens : 0,
  }
}

type PendingNondet =
  | { kind: 'clock'; value: number }
  | { kind: 'uuid';  value: string }

/**
 * RecordingIOPort — decorates an inner IOPort to emit Agent Trace events.
 *
 * Phase 3 additions:
 *  - LLM/tool events carry requestHash (cache key)
 *  - attach()/detach() emit agent.run.started/completed lifecycle events
 *  - Tool errors are recorded as structured payloads (preserve retryable/code/name)
 *
 * Phase 4 additions:
 *  - now() / uuid() record clock.read / uuid.generated events via a pending
 *    buffer that is flushed at the entry of every async method. This preserves
 *    ordering: nondet events appear in the log before the next recorded event
 *    that consumed them, without requiring async infrastructure in the sync
 *    now/uuid methods.
 */
export class RecordingIOPort implements IIOPort {
  private readonly pendingNondet: PendingNondet[] = []

  constructor(
    private readonly inner: IIOPort,
    private readonly store: IEventStore,
    private readonly runId: string,
    private readonly actor: string = 'runtime',
    private readonly objectStore?: ITraceObjectStore,
    private readonly cursor?: CausalCursor,
  ) {}

  private async outputMetadata(output: unknown): Promise<{ outputHash?: string; outputBytes?: number }> {
    let canonical: string
    try {
      canonical = canonicalize(output)
    } catch {
      return {}
    }
    if (this.objectStore) {
      try { await this.objectStore.putCanonical(canonical) } catch { /* best-effort */ }
    }
    return {
      outputHash:  contentAddressForCanonicalBytes(canonical),
      outputBytes: Buffer.byteLength(canonical, 'utf8'),
    }
  }

  /**
   * Drain pending nondet records to the store in input order. Called at
   * every async method entry so that agent-facing port.now/port.uuid
   * calls observe the invariant: nondet events appear before the next
   * recorded event that consumes them.
   *
   * Each emitted event's own `id` and `timestamp` fields use inner.uuid /
   * inner.now directly — they are infrastructure bookkeeping, not part
   * of agent-observable non-determinism, and recording them would recurse.
   */
  private async flushPendingNondet(): Promise<void> {
    while (this.pendingNondet.length > 0) {
      const item = this.pendingNondet.shift()!
      if (item.kind === 'clock') {
        await this.store.append({
          id:        this.inner.uuid(),
          runId:     this.runId,
          type:      'clock.read',
          actor:     this.actor,
          timestamp: this.inner.now(),
          payload:   { value: item.value } satisfies ClockReadPayload,
        })
      } else {
        await this.store.append({
          id:        this.inner.uuid(),
          runId:     this.runId,
          type:      'uuid.generated',
          actor:     this.actor,
          timestamp: this.inner.now(),
          payload:   { value: item.value } satisfies UuidGeneratedPayload,
        })
      }
    }
  }

  async attach(payload: AgentRunStartedPayload): Promise<void> {
    await this.flushPendingNondet()
    const id = this.inner.uuid()
    await this.store.append({
      id,
      runId:     this.runId,
      type:      'agent.run.started',
      actor:     this.actor,
      timestamp: this.inner.now(),
      payload,
    })
    // Seed the terminator so the first llm.requested can trace back to the run root.
    if (this.cursor) this.cursor.lastTerminatorId = id
  }

  async detach(payload: AgentRunCompletedPayload): Promise<void> {
    await this.flushPendingNondet()
    await this.store.append({
      id:        this.inner.uuid(),
      runId:     this.runId,
      type:      'agent.run.completed',
      actor:     this.actor,
      // The final output is produced by the last LLM response; link to it so the
      // output node can drill to the final decision (nearest-decision-ancestor).
      // causedBy is trace metadata (a bare uuid) — replay never compares it.
      ...(this.cursor?.lastLlmRespondedId ? { causedBy: this.cursor.lastLlmRespondedId } : {}),
      timestamp: this.inner.now(),
      payload,
    })
  }

  async invokeLLM(request: ModelRequest, onEvent?: (e: ModelEvent) => void): Promise<ModelResponse> {
    await this.flushPendingNondet()
    const requestHash = hashModelRequest(request)
    const reqEventId  = this.inner.uuid()
    await this.store.append({
      id:        reqEventId,
      runId:     this.runId,
      type:      'llm.requested',
      actor:     this.actor,
      // edge 2: this call was provoked by the previous turn terminator.
      ...(this.cursor?.lastTerminatorId ? { causedBy: this.cursor.lastTerminatorId } : {}),
      timestamp: this.inner.now(),
      payload:   { request, requestHash } satisfies LlmRequestedPayload,
    })
    if (this.cursor) this.cursor.lastIoEventId = reqEventId

    const response = await this.inner.invokeLLM(request, onEvent)

    const respEventId = this.inner.uuid()
    await this.store.append({
      id:        respEventId,
      runId:     this.runId,
      type:      'llm.responded',
      actor:     this.actor,
      causedBy:  reqEventId,
      timestamp: this.inner.now(),
      payload:   {
        response,
        requestHash,
        ...(cacheStatsFrom(response) ? { cacheStats: cacheStatsFrom(response) } : {}),
      } satisfies LlmRespondedPayload,
    })
    if (this.cursor) {
      this.cursor.lastLlmRespondedId = respEventId
      this.cursor.lastIoEventId      = respEventId
    }

    return response
  }

  async invokeTool(
    toolName: string,
    input: unknown,
    execute: () => Promise<unknown>,
  ): Promise<unknown> {
    await this.flushPendingNondet()
    const requestHash = hashToolCall(toolName, input)
    const reqEventId  = this.inner.uuid()
    await this.store.append({
      id:        reqEventId,
      runId:     this.runId,
      type:      'tool.requested',
      actor:     this.actor,
      // edge 1: this call was decided by the most recent llm.responded (the frame carrying toolCalls).
      ...(this.cursor?.lastLlmRespondedId ? { causedBy: this.cursor.lastLlmRespondedId } : {}),
      timestamp: this.inner.now(),
      payload:   { toolName, input, requestHash } satisfies ToolRequestedPayload,
    })
    if (this.cursor) this.cursor.lastIoEventId = reqEventId

    try {
      const output = await this.inner.invokeTool(toolName, input, execute)
      const meta = await this.outputMetadata(output)
      const respEventId = this.inner.uuid()
      await this.store.append({
        id:        respEventId,
        runId:     this.runId,
        type:      'tool.responded',
        actor:     this.actor,
        causedBy:  reqEventId,
        timestamp: this.inner.now(),
        payload:   { toolName, output, requestHash, ...meta } satisfies ToolRespondedPayload,
      })
      // tool.responded is a turn terminator for the next llm.requested (edge 2).
      // Under a parallel tool batch, several tool.responded race to write this; the
      // last-completed wins. That is intentional and harmless: any of the batch's
      // results is a valid terminator, and replay never compares trace event ids.
      if (this.cursor) {
        this.cursor.lastTerminatorId = respEventId
        this.cursor.lastIoEventId    = respEventId
      }
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

      const respEventId = this.inner.uuid()
      await this.store.append({
        id:        respEventId,
        runId:     this.runId,
        type:      'tool.responded',
        actor:     this.actor,
        causedBy:  reqEventId,
        timestamp: this.inner.now(),
        payload:   { toolName, error: errorPayload, requestHash } satisfies ToolRespondedPayload,
      })
      // An errored tool.responded still terminates the turn — the next llm.requested follows it.
      if (this.cursor) {
        this.cursor.lastTerminatorId = respEventId
        this.cursor.lastIoEventId    = respEventId
      }
      throw err
    }
  }

  now(): number {
    const value = this.inner.now()
    this.pendingNondet.push({ kind: 'clock', value })
    return value
  }

  uuid(): string {
    const value = this.inner.uuid()
    this.pendingNondet.push({ kind: 'uuid', value })
    return value
  }
}
