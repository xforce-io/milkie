export type TraceWriteStage = 'request' | 'terminal'

export interface TraceWriteErrorDetails {
  stage: TraceWriteStage
  operation: 'llm'
  eventId: string
}

const STAGE_MESSAGES: Record<TraceWriteStage, string> = {
  request:  'Failed to append LLM request event to the trace.',
  terminal: 'Failed to append LLM terminal event to the trace.',
}

/**
 * Live-only failure when EventStore.append rejects for an LLM request or
 * terminal. Cause may be retained for process logging but is never serialized
 * into Trace / AgentResult / Replay outcomes.
 */
export class TraceWriteError extends Error {
  readonly stage: TraceWriteStage
  readonly operation = 'llm' as const
  readonly eventId: string
  override readonly cause?: unknown

  constructor(details: TraceWriteErrorDetails, cause?: unknown) {
    super(STAGE_MESSAGES[details.stage])
    this.name = 'TraceWriteError'
    this.stage = details.stage
    this.eventId = details.eventId
    this.cause = cause
  }
}
