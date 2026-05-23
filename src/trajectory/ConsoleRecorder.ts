import { v4 as uuid } from 'uuid'
import type { ITrajectoryRecorder, Span, SpanAttributes } from '../types/trajectory.js'

export class ConsoleRecorder implements ITrajectoryRecorder {
  readonly traceId: string

  constructor(traceId?: string) {
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
    console.log(`[SPAN START] ${name}`, attributes)
    return span
  }

  endSpan(span: Span, status: 'ok' | 'error' = 'ok'): void {
    span.endTime = Date.now()
    span.status  = status
    const duration = span.endTime - span.startTime
    console.log(`[SPAN END]   ${span.name} (${status}, ${duration}ms)`)
  }

  recordEvent(span: Span, name: string, attributes: SpanAttributes = {}): void {
    span.events.push({ name, timestamp: Date.now(), attributes })
    console.log(`[EVENT]      ${span.name} → ${name}`, attributes)
  }

  async flush(): Promise<void> {}
}
