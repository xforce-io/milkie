import { v4 as uuid } from 'uuid'
import type { ITrajectoryRecorder, Span, SpanAttributes, Trajectory } from '../types/trajectory.js'

export class InMemoryRecorder implements ITrajectoryRecorder {
  private spans: Map<string, Span> = new Map()
  readonly traceId: string

  constructor(
    traceId?: string,
    private readonly agentId: string = '',
  ) {
    this.traceId = traceId ?? uuid()
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
    this.spans.set(span.spanId, span)
    return span
  }

  endSpan(span: Span, status: 'ok' | 'error' = 'ok'): void {
    span.endTime = Date.now()
    span.status  = status
  }

  recordEvent(span: Span, name: string, attributes: SpanAttributes = {}): void {
    span.events.push({ name, timestamp: Date.now(), attributes })
  }

  async flush(): Promise<void> {}

  getTrajectory(status: Trajectory['status'] = 'running'): Trajectory {
    const allSpans = Array.from(this.spans.values())
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
    return Array.from(this.spans.values())
  }
}
