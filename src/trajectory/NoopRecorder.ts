import type { ITrajectoryRecorder, Span, SpanAttributes } from '../types/trajectory.js'

export class NoopRecorder implements ITrajectoryRecorder {
  startSpan(_name: string, _attributes?: SpanAttributes, _parentSpanId?: string): Span {
    return {
      spanId: '',
      traceId: '',
      name: _name,
      startTime: Date.now(),
      attributes: {},
      events: [],
    }
  }

  endSpan(_span: Span, _status?: 'ok' | 'error'): void {}
  recordEvent(_span: Span, _name: string, _attributes?: SpanAttributes): void {}
  async flush(): Promise<void> {}
}
