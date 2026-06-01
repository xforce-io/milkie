export type SpanAttributes = Record<string, unknown>

export interface Span {
  spanId:      string
  traceId:     string
  parentSpanId?: string
  name:        string
  startTime:   number
  endTime?:    number
  status?:     'ok' | 'error'
  attributes:  SpanAttributes
  events:      SpanEvent[]
}

export interface SpanEvent {
  name:       string
  timestamp:  number
  attributes: SpanAttributes
}

export interface ITrajectoryRecorder {
  startSpan(name: string, attributes?: SpanAttributes, parentSpanId?: string): Span
  endSpan(span: Span, status?: 'ok' | 'error'): void
  recordEvent(span: Span, name: string, attributes?: SpanAttributes): void
  flush(): Promise<void>
}

export interface ResolvedManifest {
  agentId:      string
  agentVersion: string
  model?: {
    provider: string
    model:    string
    adapter:  string
    baseUrl?: string
  }
  tools:      Record<string, { version: string }>
  toolboxes:  Record<string, { version: string }>
  skills:     Record<string, { version: string }>
  subAgents:  Record<string, { version: string }>
}

export interface Trajectory {
  traceId:   string
  agentId:   string
  startTime: number
  endTime?:  number
  status:    'running' | 'completed' | 'interrupted' | 'failed'
  spans:     Span[]
  resolvedManifest?: ResolvedManifest
  metrics?: {
    totalTokens: number
    totalSteps:  number
    duration:    number
    cost:        number
  }
}
