import type {
  Event,
  ToolRespondedPayload,
  ClockReadPayload,
  UuidGeneratedPayload,
} from './types.js'
import { LLM_OUTCOME_SCHEMA_VERSION } from './types.js'
import type { ModelResponse } from '../types/model.js'
import {
  decodeLlmOutcome,
  reconstructLlmError,
  type LlmOutcome,
} from './LlmOutcome.js'
import { TraceIntegrityError } from './TraceIntegrityError.js'

interface IndexedOutcome {
  outcome: LlmOutcome
  /** Request append index for paired outcomes; terminal index for terminal-only legacy. */
  order: number
}

interface ToolOutcome {
  output?: unknown
  error?:  ToolRespondedPayload['error']
}

/**
 * In-memory projection of LLM/tool response events keyed by canonical
 * request hash, with one FIFO queue per hash. LLM queues hold LlmOutcome
 * (success or failure). Drives strict structural replay.
 */
export class CacheIndex {
  private readonly llm:   Map<string, LlmOutcome[]>
  private readonly tool:  Map<string, ToolOutcome[]>
  private readonly clock: number[]
  private readonly uuid:  string[]

  private constructor(
    llm:   Map<string, LlmOutcome[]>,
    tool:  Map<string, ToolOutcome[]>,
    clock: number[],
    uuid:  string[],
  ) {
    this.llm   = llm
    this.tool  = tool
    this.clock = clock
    this.uuid  = uuid
  }

  /**
   * Build a CacheIndex from a run's events. Validates LLM request/terminal
   * integrity (§8.6) before enqueueing. Throws TraceIntegrityError on corruption.
   */
  static fromEvents(events: Event[]): CacheIndex {
    // 1. Global event-id uniqueness.
    const seenIds = new Map<string, number>()
    for (let i = 0; i < events.length; i++) {
      const ev = events[i]!
      if (typeof ev.id !== 'string' || ev.id.length === 0) {
        throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: String(ev.id ?? '') })
      }
      if (seenIds.has(ev.id)) {
        throw new TraceIntegrityError({ kind: 'duplicate_event_id', eventId: ev.id })
      }
      seenIds.set(ev.id, i)
    }

    type ReqInfo = {
      index: number
      id: string
      hash: string
      isV2: boolean
      hasHash: boolean
      claimedBy?: string
    }
    type TermInfo = {
      index: number
      event: Event
      hash: string
      causedBy?: string
      isLegacy: boolean
      outcome: LlmOutcome
    }

    const requests: ReqInfo[] = []
    const requestById = new Map<string, ReqInfo>()
    const terminals: TermInfo[] = []

    for (let i = 0; i < events.length; i++) {
      const ev = events[i]!
      if (ev.type === 'llm.requested') {
        const p = ev.payload as Record<string, unknown> | undefined
        if (!p || typeof p !== 'object' || Array.isArray(p)) {
          throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: ev.id })
        }
        const hasOutcomeSchemaVersion = Object.prototype.hasOwnProperty.call(p, 'outcomeSchemaVersion')
        if (hasOutcomeSchemaVersion) {
          // Any new-format marker must be exactly v2 with a non-empty requestHash.
          // Never treat a broken v2 request as Phase-2 skip (would become divergence).
          if (p.outcomeSchemaVersion !== LLM_OUTCOME_SCHEMA_VERSION) {
            throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: ev.id })
          }
          if (typeof p.requestHash !== 'string' || p.requestHash.length === 0) {
            throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: ev.id })
          }
          if (!p.request || typeof p.request !== 'object' || Array.isArray(p.request)) {
            throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: ev.id })
          }
          const info: ReqInfo = {
            index: i,
            id: ev.id,
            hash: p.requestHash,
            isV2: true,
            hasHash: true,
          }
          requests.push(info)
          requestById.set(ev.id, info)
          continue
        }

        const hash = typeof p.requestHash === 'string' ? p.requestHash : ''
        // Phase 2 / legacy: empty hash is not replay-indexable; skip pairing.
        const hasHash = hash.length > 0
        const info: ReqInfo = { index: i, id: ev.id, hash, isV2: false, hasHash }
        requests.push(info)
        requestById.set(ev.id, info)
      } else if (ev.type === 'llm.responded') {
        // causedBy if present must be non-empty string
        if (ev.causedBy !== undefined && (typeof ev.causedBy !== 'string' || ev.causedBy.length === 0)) {
          throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: ev.id })
        }
        const payload = ev.payload as Record<string, unknown> | undefined
        // Phase 2 no-hash terminals: skip (not replay-indexable), but only when
        // they also lack status (legacy/phase2). A v2 terminal without hash is malformed.
        if (payload && typeof payload === 'object'
          && (payload.requestHash === undefined || payload.requestHash === '')
          && payload.status === undefined) {
          continue
        }
        let outcome: LlmOutcome
        try {
          outcome = decodeLlmOutcome(ev)
        } catch (err) {
          if (err instanceof TraceIntegrityError) throw err
          throw new TraceIntegrityError({ kind: 'malformed_payload', eventId: ev.id })
        }
        terminals.push({
          index: i,
          event: ev,
          hash: outcome.requestHash,
          causedBy: typeof ev.causedBy === 'string' ? ev.causedBy : undefined,
          isLegacy: outcome.status === 'ok' && outcome.legacy === true,
          outcome,
        })
      }
    }

    const indexed: IndexedOutcome[] = []
    // Per-hash FIFO of unclaimed legacy requests (for no-causedBy pairing).
    const legacyUnclaimedByHash = new Map<string, ReqInfo[]>()
    const v2Hashes = new Set<string>()
    const legacyHashes = new Set<string>()

    for (const req of requests) {
      if (!req.hasHash) continue
      if (req.isV2) v2Hashes.add(req.hash)
      else {
        legacyHashes.add(req.hash)
        const q = legacyUnclaimedByHash.get(req.hash)
        if (q) q.push(req)
        else legacyUnclaimedByHash.set(req.hash, [req])
      }
    }

    for (const term of terminals) {
      if (term.causedBy !== undefined) {
        const req = requestById.get(term.causedBy)
        if (!req) {
          throw new TraceIntegrityError({
            kind: 'orphan_terminal',
            eventId: term.event.id,
            requestEventId: term.causedBy,
          })
        }
        if (req.claimedBy !== undefined) {
          throw new TraceIntegrityError({
            kind: 'duplicate_terminal',
            eventId: term.event.id,
            requestEventId: req.id,
          })
        }
        if (!req.hasHash) {
          // Phase 2 request paired somehow — treat as orphan for replay index.
          throw new TraceIntegrityError({
            kind: 'orphan_terminal',
            eventId: term.event.id,
            requestEventId: req.id,
          })
        }
        if (req.hash !== term.hash) {
          throw new TraceIntegrityError({
            kind: 'hash_mismatch',
            eventId: term.event.id,
            requestEventId: req.id,
          })
        }
        // v2 request only accepts v2 terminal (status present → not legacy)
        if (req.isV2 && term.isLegacy) {
          throw new TraceIntegrityError({
            kind: 'malformed_payload',
            eventId: term.event.id,
            requestEventId: req.id,
          })
        }
        // legacy request only accepts legacy terminal
        if (!req.isV2 && !term.isLegacy) {
          throw new TraceIntegrityError({
            kind: 'malformed_payload',
            eventId: term.event.id,
            requestEventId: req.id,
          })
        }
        req.claimedBy = term.event.id
        // Remove from legacy unclaimed FIFO if present
        if (!req.isV2) {
          const q = legacyUnclaimedByHash.get(req.hash)
          if (q) {
            const idx = q.findIndex(r => r.id === req.id)
            if (idx >= 0) q.splice(idx, 1)
          }
        }
        indexed.push({ outcome: term.outcome, order: req.index })
        continue
      }

      // No causedBy — only legal for legacy terminals.
      if (!term.isLegacy) {
        throw new TraceIntegrityError({
          kind: 'malformed_payload',
          eventId: term.event.id,
        })
      }

      const legacyQ = legacyUnclaimedByHash.get(term.hash)
      if (legacyQ && legacyQ.length > 0) {
        const req = legacyQ.shift()!
        req.claimedBy = term.event.id
        indexed.push({ outcome: term.outcome, order: req.index })
        continue
      }

      // No unclaimed legacy request for this hash.
      if (v2Hashes.has(term.hash)) {
        // Ambiguous: v2 requests exist for this hash and terminal lacks causedBy.
        throw new TraceIntegrityError({
          kind: 'ambiguous_legacy',
          eventId: term.event.id,
        })
      }

      if (legacyHashes.has(term.hash)) {
        // Legacy requests existed but all already claimed → duplicate_terminal.
        throw new TraceIntegrityError({
          kind: 'duplicate_terminal',
          eventId: term.event.id,
        })
      }

      // Terminal-only legacy success (old Phase 3): no request of this hash at all.
      indexed.push({ outcome: term.outcome, order: term.index })
    }

    // Dangling requests: any request with hash that never got a terminal.
    for (const req of requests) {
      if (!req.hasHash) continue
      if (req.claimedBy === undefined) {
        throw new TraceIntegrityError({
          kind: 'dangling_request',
          eventId: req.id,
          requestEventId: req.id,
        })
      }
    }

    // Sort outcomes by order anchor, then group into per-hash FIFO.
    indexed.sort((a, b) => a.order - b.order)
    const llm: Map<string, LlmOutcome[]> = new Map()
    for (const item of indexed) {
      push(llm, item.outcome.requestHash, item.outcome)
    }

    // Tool / clock / uuid — unchanged semantics.
    const tool:  Map<string, ToolOutcome[]> = new Map()
    const clock: number[] = []
    const uuid:  string[] = []

    for (const ev of events) {
      if (ev.type === 'tool.responded') {
        const p = ev.payload as ToolRespondedPayload
        if (!p.requestHash) continue
        push(tool, p.requestHash, { output: p.output, error: p.error })
      } else if (ev.type === 'clock.read') {
        clock.push((ev.payload as ClockReadPayload).value)
      } else if (ev.type === 'uuid.generated') {
        uuid.push((ev.payload as UuidGeneratedPayload).value)
      }
    }

    return new CacheIndex(llm, tool, clock, uuid)
  }

  /**
   * Consume the next LlmOutcome for `hash`. Success returns the response;
   * failure throws the reconstructed typed error. Empty queue → CacheIndexEmptyError.
   */
  consumeLLM(hash: string): ModelResponse {
    const q = this.llm.get(hash)
    if (!q || q.length === 0) throw new CacheIndexEmptyError(`CacheIndex: LLM queue empty for hash ${hash}`)
    const outcome = q.shift()!
    if (outcome.status === 'ok') return outcome.response
    throw reconstructLlmError(outcome.error)
  }

  /** Peek remaining LLM outcomes without consuming (test/diagnostics helper). */
  peekLLM(hash: string): LlmOutcome[] {
    return [...(this.llm.get(hash) ?? [])]
  }

  consumeTool(hash: string): unknown {
    const q = this.tool.get(hash)
    if (!q || q.length === 0) throw new CacheIndexEmptyError(`CacheIndex: tool queue empty for hash ${hash}`)
    const outcome = q.shift()!
    if (outcome.error) {
      const err = new Error(outcome.error.message) as Error & { retryable?: boolean; code?: string }
      if (outcome.error.retryable !== undefined) err.retryable = outcome.error.retryable
      if (outcome.error.code !== undefined)      err.code      = outcome.error.code
      if (outcome.error.name !== undefined)      err.name      = outcome.error.name
      throw err
    }
    return outcome.output
  }

  consumeClock(): number {
    if (this.clock.length === 0) throw new CacheIndexEmptyError('CacheIndex: clock queue empty')
    return this.clock.shift()!
  }

  consumeUuid(): string {
    if (this.uuid.length === 0) throw new CacheIndexEmptyError('CacheIndex: uuid queue empty')
    return this.uuid.shift()!
  }

  remaining(): { llm: number; tool: number; clock: number; uuid: number } {
    let llmCount = 0, toolCount = 0
    for (const q of this.llm.values())  llmCount  += q.length
    for (const q of this.tool.values()) toolCount += q.length
    return { llm: llmCount, tool: toolCount, clock: this.clock.length, uuid: this.uuid.length }
  }

  allHashes(): { llm: string[]; tool: string[] } {
    return { llm: [...this.llm.keys()], tool: [...this.tool.keys()] }
  }
}

function push<K, V>(map: Map<K, V[]>, key: K, value: V): void {
  const q = map.get(key)
  if (q) q.push(value)
  else   map.set(key, [value])
}

/**
 * Thrown by CacheIndex.consumeLLM / consumeTool when the FIFO queue for a
 * given hash is empty (i.e. the replay has consumed all recorded responses).
 * Using a named class lets callers distinguish "queue exhausted" from a
 * reconstructed tool/LLM error (which is also an Error) without fragile
 * message-prefix matching.
 */
export class CacheIndexEmptyError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CacheIndexEmptyError'
  }
}
