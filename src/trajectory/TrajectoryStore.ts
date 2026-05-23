import { v4 as uuid } from 'uuid'
import fs from 'fs'
import path from 'path'
import type { ITrajectoryRecorder, ResolvedManifest, Span, SpanAttributes, Trajectory } from '../types/trajectory.js'

export interface SpanMeta {
  agentRunId: string
  contextId:  string
  agentId:    string
  resolvedManifest?: ResolvedManifest
}

interface StoredSpan {
  span: Span
  meta: SpanMeta
}

/**
 * Queryable in-memory store that collects spans from multiple recorders.
 * Allows querying by agentRunId, contextId, or agentId.
 */
export class TrajectoryStore {
  private readonly entries: StoredSpan[] = []
  private readonly manifests: Map<string, ResolvedManifest> = new Map()

  constructor(private readonly opts: { jsonlDir?: string } = {}) {}

  record(span: Span, meta: SpanMeta): void {
    this.entries.push({ span, meta })
    if (meta.resolvedManifest) {
      this.manifests.set(meta.agentRunId, meta.resolvedManifest)
    }
  }

  async getByRunId(agentRunId: string): Promise<Trajectory> {
    const spans = this.entries
      .filter(e => e.meta.agentRunId === agentRunId)
      .map(e => e.span)

    const agentId = this.entries.find(e => e.meta.agentRunId === agentRunId)?.meta.agentId ?? ''
    return this.buildTrajectory(spans, agentId, this.manifests.get(agentRunId))
  }

  async getByContextId(contextId: string): Promise<Trajectory> {
    const spans = this.entries
      .filter(e => e.meta.contextId === contextId)
      .map(e => e.span)

    const agentId = this.entries.find(e => e.meta.contextId === contextId)?.meta.agentId ?? ''
    return this.buildTrajectory(spans, agentId, this.entries.find(e => e.meta.contextId === contextId)?.meta.resolvedManifest)
  }

  async getByAgentId(agentId: string): Promise<Trajectory> {
    const spans = this.entries
      .filter(e => e.meta.agentId === agentId)
      .map(e => e.span)

    return this.buildTrajectory(spans, agentId, this.entries.find(e => e.meta.agentId === agentId)?.meta.resolvedManifest)
  }

  /**
   * Returns a recorder whose spans are auto-stored in this store.
   */
  makeRecorder(
    meta: SpanMeta & { traceId?: string },
  ): ITrajectoryRecorder {
    return new StoringRecorder(this, meta, this.opts.jsonlDir)
  }

  private buildTrajectory(spans: Span[], agentId: string, resolvedManifest?: ResolvedManifest): Trajectory {
    const startTime = spans.length > 0
      ? Math.min(...spans.map(s => s.startTime))
      : Date.now()

    const endTime = spans.every(s => s.endTime !== undefined)
      ? Math.max(...spans.map(s => s.endTime ?? 0))
      : undefined

    const hasError = spans.some(s => s.status === 'error')
    const status: Trajectory['status'] = endTime === undefined
      ? 'running'
      : hasError ? 'failed' : 'completed'

    const traceId = spans[0]?.traceId ?? uuid()

    return {
      traceId,
      agentId,
      startTime,
      endTime,
      status,
      spans,
      resolvedManifest,
    }
  }
}

/**
 * InMemoryRecorder that also forwards each ended span to the parent TrajectoryStore.
 */
class StoringRecorder implements ITrajectoryRecorder {
  private readonly localSpans: Map<string, Span> = new Map()
  readonly traceId: string
  private readonly agentId: string
  private readonly meta: SpanMeta

  constructor(
    private readonly store: TrajectoryStore,
    metaWithTrace: SpanMeta & { traceId?: string },
    private readonly jsonlDir?: string,
  ) {
    this.traceId = metaWithTrace.traceId ?? uuid()
    this.agentId = metaWithTrace.agentId
    this.meta    = {
      agentRunId: metaWithTrace.agentRunId,
      contextId:  metaWithTrace.contextId,
      agentId:    metaWithTrace.agentId,
      resolvedManifest: metaWithTrace.resolvedManifest,
    }
    if (this.jsonlDir) fs.mkdirSync(this.jsonlDir, { recursive: true })
  }

  startSpan(name: string, attributes: SpanAttributes = {}, parentSpanId?: string): Span {
    const span: Span = {
      spanId:      uuid(),
      traceId:     this.traceId,
      parentSpanId,
      name,
      startTime:   Date.now(),
      attributes,
      events:      [],
    }
    this.localSpans.set(span.spanId, span)
    return span
  }

  endSpan(span: Span, status: 'ok' | 'error' = 'ok'): void {
    span.endTime = Date.now()
    span.status  = status
    // Forward to the store so it is queryable
    this.store.record(span, this.meta)
    if (this.jsonlDir) {
      const file = path.join(this.jsonlDir, `${this.meta.agentId}-${this.traceId}.jsonl`)
      fs.appendFileSync(file, JSON.stringify({ meta: this.meta, span }) + '\n')
    }
  }

  recordEvent(span: Span, name: string, attributes: SpanAttributes = {}): void {
    span.events.push({ name, timestamp: Date.now(), attributes })
  }

  async flush(): Promise<void> {}

  /** Convenience: build a Trajectory from spans recorded so far */
  getTrajectory(status: Trajectory['status'] = 'running'): Trajectory {
    const allSpans = Array.from(this.localSpans.values())
    return {
      traceId:   this.traceId,
      agentId:   this.agentId,
      startTime: allSpans[0]?.startTime ?? Date.now(),
      endTime:   status !== 'running' ? Date.now() : undefined,
      status,
      spans:     allSpans,
    }
  }

  getSpans(): Span[] {
    return Array.from(this.localSpans.values())
  }
}
