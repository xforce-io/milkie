import type { ModelRequest, ModelResponse } from '../types/model.js'
import {
  assertToolInvocationAllowed,
  resolveIOInvocationControl,
  type IIOPort,
  type LLMInvocationOptions,
  type ToolInvocationOptions,
} from '../runtime/IOPort.js'
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
  ObjectCreatedPayload,
  RelationCreatedPayload,
  LineageBuffer,
  TrustedProviderFamily,
} from './types.js'
import { LLM_OUTCOME_SCHEMA_VERSION } from './types.js'
import { hashModelRequest, hashToolCall, canonicalize, contentAddressForCanonicalBytes } from './hash.js'
import type { ITraceObjectStore } from './TraceObjectStore.js'
import type { CausalCursor } from './CausalCursor.js'
import { sanitizeModelRequestForTrace } from './imageSummary.js'
import {
  buildFailureTerminalPayload,
  buildSuccessTerminalPayload,
  normalizeLlmFailure,
  reconstructLlmError,
  type TrustedProviderContext,
} from './LlmOutcome.js'
import { TraceWriteError } from './TraceWriteError.js'

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
    private readonly trustedProvider?: TrustedProviderFamily | 'unknown',
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
    // Prefer lastLlmTerminalId so error completions link to the failure terminal;
    // fall back to lastLlmRespondedId for older cursors / success-only paths.
    const completionCause =
      this.cursor?.lastLlmTerminalId ?? this.cursor?.lastLlmRespondedId
    await this.store.append({
      id:        this.inner.uuid(),
      runId:     this.runId,
      type:      'agent.run.completed',
      actor:     this.actor,
      // The final output / error terminal; link so the output node can drill to it.
      // causedBy is trace metadata (a bare uuid) — replay never compares it.
      ...(completionCause ? { causedBy: completionCause } : {}),
      timestamp: this.inner.now(),
      payload,
    })
  }

  async invokeLLM(request: ModelRequest, options?: LLMInvocationOptions): Promise<ModelResponse> {
    // 0. Shared #228 control resolve first — invalid rejects before any request/inner.
    const control = resolveIOInvocationControl(options?.control)
    const resolvedOptions = control ? { ...options, control } : options
    await this.flushPendingNondet()

    // Hash the live request (including base64) so record/replay keys match.
    // Persist a redacted copy that never stores inline media bytes or URL secrets.
    const requestHash = hashModelRequest(request)
    const safeRequest = sanitizeModelRequestForTrace(request)
    const reqEventId  = this.inner.uuid()
    const trustedContext: TrustedProviderContext | undefined =
      this.trustedProvider !== undefined
        ? { providerFamily: this.trustedProvider }
        : undefined

    // 1. v2 request append — fail closed (no inner call on rejection).
    try {
      await this.store.append({
        id:        reqEventId,
        runId:     this.runId,
        type:      'llm.requested',
        actor:     this.actor,
        // edge 2: this call was provoked by the previous turn terminator.
        ...(this.cursor?.lastTerminatorId ? { causedBy: this.cursor.lastTerminatorId } : {}),
        timestamp: this.inner.now(),
        payload:   {
          request: safeRequest as ModelRequest,
          requestHash,
          outcomeSchemaVersion: LLM_OUTCOME_SCHEMA_VERSION,
        } satisfies LlmRequestedPayload,
      })
    } catch (err) {
      throw new TraceWriteError({ stage: 'request', operation: 'llm', eventId: reqEventId }, err)
    }
    if (this.cursor) this.cursor.lastIoEventId = reqEventId

    // 2. Capture inner outcome in memory (never rethrow original untrusted error).
    let terminalPayload: LlmRespondedPayload
    let successResponse: ModelResponse | undefined
    try {
      const response = await this.inner.invokeLLM(request, resolvedOptions)
      successResponse = response
      terminalPayload = buildSuccessTerminalPayload(
        response,
        requestHash,
        cacheStatsFrom(response),
      )
    } catch (err) {
      const envelope = normalizeLlmFailure(err, request, trustedContext)
      terminalPayload = buildFailureTerminalPayload(envelope, requestHash)
    }

    // 3. Exactly one terminal append. Rejection → TraceWriteError; no second append.
    const respEventId = this.inner.uuid()
    try {
      await this.store.append({
        id:        respEventId,
        runId:     this.runId,
        type:      'llm.responded',
        actor:     this.actor,
        causedBy:  reqEventId,
        timestamp: this.inner.now(),
        payload:   terminalPayload,
      })
    } catch (err) {
      throw new TraceWriteError({ stage: 'terminal', operation: 'llm', eventId: respEventId }, err)
    }

    // 4. Cursor update from the written terminal, then return/rebuild.
    if (this.cursor) {
      this.cursor.lastLlmTerminalId = respEventId
      this.cursor.lastIoEventId = respEventId
      if (terminalPayload.status === 'ok') {
        this.cursor.lastLlmRespondedId = respEventId
      }
    }

    if (terminalPayload.status === 'ok') {
      return successResponse!
    }
    // Rebuild typed error from the written envelope so live/Trace/Replay share identity.
    throw reconstructLlmError(terminalPayload.error)
  }

  /**
   * #37/#38: drain the lineage buffer the handler filled, as object.created /
   * relation.created events anchored to the just-written tool.responded
   * (`producerEventId` / `causedByEventId` = respEventId). Each event's own id is
   * infrastructure bookkeeping (inner.uuid), like the nondet flush above.
   */
  private async flushLineage(lineage: LineageBuffer | undefined, producerEventId: string): Promise<void> {
    if (!lineage) return
    for (const o of lineage.objects) {
      // #160: prefer the object's own producerEventId (backfilled from the retrieval turn).
      const effectiveProducerEventId = o.producerEventId ?? producerEventId
      await this.store.append({
        id:        this.inner.uuid(),
        runId:     this.runId,
        type:      'object.created',
        actor:     this.actor,
        causedBy:  effectiveProducerEventId,
        timestamp: this.inner.now(),
        payload:   { objectId: o.objectId, type: o.type, producerEventId: effectiveProducerEventId, ...(o.hash ? { hash: o.hash } : {}), ...(o.meta ? { meta: o.meta } : {}) } satisfies ObjectCreatedPayload,
      })
    }
    for (const r of lineage.relations) {
      await this.store.append({
        id:        this.inner.uuid(),
        runId:     this.runId,
        type:      'relation.created',
        actor:     this.actor,
        causedBy:  producerEventId,
        timestamp: this.inner.now(),
        payload:   { relationId: r.relationId, type: r.type, fromObjectId: r.fromObjectId, toObjectId: r.toObjectId, causedByEventId: producerEventId, ...(r.meta ? { meta: r.meta } : {}) } satisfies RelationCreatedPayload,
      })
    }
  }

  async invokeTool(
    toolName: string,
    input: unknown,
    execute: (signal: AbortSignal) => Promise<unknown>,
    opts?: ToolInvocationOptions,
  ): Promise<unknown> {
    const control = resolveIOInvocationControl(opts?.control)
    const resolvedOptions = control ? { ...opts, control } : opts
    await this.flushPendingNondet()
    const requestHash = hashToolCall(toolName, input, opts?.invalidArguments)
    // #81: only stamp toolCallId when supplied, so id-less callers stay clean and
    // old traces (no id) read back identically.
    const idField = opts?.toolCallId ? { toolCallId: opts.toolCallId } : {}
    const reqEventId  = this.inner.uuid()
    await this.store.append({
      id:        reqEventId,
      runId:     this.runId,
      type:      'tool.requested',
      actor:     this.actor,
      // edge 1: this call was decided by the most recent llm.responded (the frame carrying toolCalls).
      ...(this.cursor?.lastLlmRespondedId ? { causedBy: this.cursor.lastLlmRespondedId } : {}),
      timestamp: this.inner.now(),
      payload:   { toolName, ...idField, input, requestHash, ...(opts?.invalidArguments ? { invalidArguments: opts.invalidArguments } : {}) } satisfies ToolRequestedPayload,
    })
    if (this.cursor) this.cursor.lastIoEventId = reqEventId

    try {
      // #237: only fail-closed with RUN_* when control already carries an
      // adjudicated reason. Runtime passes a stable signal-only snapshot for
      // effect abort propagation; pure signal/deadline effect control stays
      // #228 IO_* on the inner port, and AgentRuntime maps in-flight
      // IOControlError → RUN_* when RunControl has stopped. Calling the RUN
      // gate on a bare aborted signal would mis-label deadline as cancelled.
      if (control?.reason === 'deadline' || control?.reason === 'cancelled') {
        assertToolInvocationAllowed(control, this.nowSample())
      }
      const output = await this.inner.invokeTool(toolName, input, execute, resolvedOptions)

      const meta = await this.outputMetadata(output)
      // #37: the handler may have declared objects during execute; list their ids
      // on tool.responded (artifactRefs) and emit object.created/relation.created.
      const artifactRefs = opts?.lineage?.objects.map(o => o.objectId)
      const respEventId = this.inner.uuid()
      // #160: backfill producerEventId on lazily-registered objects before flushing.
      opts?.lineage?.backfillProducerEventId?.(respEventId)
      await this.store.append({
        id:        respEventId,
        runId:     this.runId,
        type:      'tool.responded',
        actor:     this.actor,
        causedBy:  reqEventId,
        timestamp: this.inner.now(),
        payload:   { toolName, ...idField, status: 'ok', output, requestHash, ...meta, ...(artifactRefs && artifactRefs.length ? { artifactRefs } : {}) } satisfies ToolRespondedPayload,
      })
      await this.flushLineage(opts?.lineage, respEventId)
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
        payload:   { toolName, ...idField, status: 'error', error: errorPayload, requestHash, ...(opts?.invalidArguments ? { invalidArguments: opts.invalidArguments } : {}) } satisfies ToolRespondedPayload,
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

  /**
   * Infrastructure clock for run-control gates/timers. Shares inner timeline
   * without enqueueing clock.read (not agent-observable nondet).
   */
  nowSample(): number {
    return this.inner.nowSample?.() ?? this.inner.now()
  }

  uuid(): string {
    const value = this.inner.uuid()
    this.pendingNondet.push({ kind: 'uuid', value })
    return value
  }
}
