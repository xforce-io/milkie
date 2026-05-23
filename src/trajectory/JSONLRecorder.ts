import fs from 'fs'
import path from 'path'
import { v4 as uuid } from 'uuid'
import type { ITrajectoryRecorder, Span, SpanAttributes } from '../types/trajectory.js'

export interface JSONLRecorderOptions {
  dir:     string
  agentId: string
  traceId?: string
}

export class JSONLRecorder implements ITrajectoryRecorder {
  private readonly fd:      number
  private readonly traceId: string
  private spans: Map<string, Span> = new Map()

  constructor(private readonly opts: JSONLRecorderOptions) {
    this.traceId = opts.traceId ?? uuid()
    fs.mkdirSync(opts.dir, { recursive: true })
    const file = path.join(opts.dir, `${opts.agentId}-${this.traceId}.jsonl`)
    this.fd = fs.openSync(file, 'a')
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
    this.writeLine({ ...span })
  }

  recordEvent(span: Span, name: string, attributes: SpanAttributes = {}): void {
    span.events.push({ name, timestamp: Date.now(), attributes })
  }

  async flush(): Promise<void> {
    fs.fdatasyncSync(this.fd)
  }

  close(): void {
    fs.closeSync(this.fd)
  }

  private writeLine(data: unknown): void {
    fs.writeSync(this.fd, JSON.stringify(data) + '\n')
  }
}
