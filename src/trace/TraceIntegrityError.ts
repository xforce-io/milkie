export type TraceIntegrityErrorKind =
  | 'duplicate_event_id'
  | 'duplicate_terminal'
  | 'orphan_terminal'
  | 'hash_mismatch'
  | 'dangling_request'
  | 'malformed_payload'
  | 'ambiguous_legacy'

export interface TraceIntegrityErrorDetails {
  kind: TraceIntegrityErrorKind
  eventId?: string
  requestEventId?: string
}

const KIND_MESSAGES: Record<TraceIntegrityErrorKind, string> = {
  duplicate_event_id:  'Trace integrity error: duplicate event id.',
  duplicate_terminal:  'Trace integrity error: duplicate LLM terminal.',
  orphan_terminal:     'Trace integrity error: orphan LLM terminal.',
  hash_mismatch:       'Trace integrity error: LLM request/terminal hash mismatch.',
  dangling_request:    'Trace integrity error: dangling LLM request.',
  malformed_payload:   'Trace integrity error: malformed LLM payload.',
  ambiguous_legacy:    'Trace integrity error: ambiguous legacy LLM terminal.',
}

/**
 * Thrown while building CacheIndex when persisted events violate the LLM
 * outcome integrity rules. Carries only kind + event identifiers — never payload.
 */
export class TraceIntegrityError extends Error {
  readonly kind: TraceIntegrityErrorKind
  readonly eventId?: string
  readonly requestEventId?: string

  constructor(details: TraceIntegrityErrorDetails) {
    super(KIND_MESSAGES[details.kind])
    this.name = 'TraceIntegrityError'
    this.kind = details.kind
    if (details.eventId !== undefined) this.eventId = details.eventId
    if (details.requestEventId !== undefined) this.requestEventId = details.requestEventId
  }
}
