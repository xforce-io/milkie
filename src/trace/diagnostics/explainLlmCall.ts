import type { Event, EventKind } from '../types.js'
import { walkCausedBy } from './walkCausedBy.js'
import { summarizeEvent } from './summarizeEvent.js'
import { fsmStateAt } from './fsmStateAt.js'
import { contextRefsAt } from '../RegionContextView.js'

export interface LlmCallExplanation {
  llmRequestedEventId: string
  trigger: {
    causedByEventId?: string
    causedBySummary?: string
  }
  fsmState: string | null
  regionCount: number
  causalChain: Array<{ eventId: string; type: EventKind; summary: string }>
  summary: string
}

/**
 * Pure event-log projection: explain why a given llm.requested fired — the
 * turn-terminator that triggered it, the FSM state at the time, how many
 * regions composed the prompt, and the causal chain. No LLM, no I/O, no
 * stored snapshot. Returns a plain serializable object (the JSON shape the
 * CLI explainer #36 will emit). Region details live in #26's "Assembled by".
 */
export function explainLlmCall(events: Event[], llmRequestedEventId: string): LlmCallExplanation {
  const byId = new Map<string, Event>()
  for (const e of events) byId.set(e.id, e)

  const evt = byId.get(llmRequestedEventId)
  if (!evt) throw new Error(`explainLlmCall: unknown event id "${llmRequestedEventId}"`)
  if (evt.type !== 'llm.requested') {
    throw new Error(`explainLlmCall: event "${llmRequestedEventId}" is "${evt.type}", expected "llm.requested"`)
  }

  const causeEvt = evt.causedBy ? byId.get(evt.causedBy) : undefined
  const causeSummary = causeEvt ? summarizeEvent(causeEvt) : undefined
  const fsmState = fsmStateAt(events, llmRequestedEventId)
  const regionCount = contextRefsAt(events, llmRequestedEventId, 'at').size

  const causalChain = walkCausedBy(events, llmRequestedEventId).map(e => ({
    eventId: e.id,
    type:    e.type,
    summary: summarizeEvent(e),
  }))

  const triggerSource = causeSummary ?? '(无上游记录)'
  const summary = `LLM 调用 @ state ${fsmState ?? '?'},由 ${triggerSource} 触发;prompt 由 ${regionCount} 个 region 拼成`

  return {
    llmRequestedEventId,
    trigger: {
      ...(evt.causedBy ? { causedByEventId: evt.causedBy } : {}),
      ...(causeSummary ? { causedBySummary: causeSummary } : {}),
    },
    fsmState,
    regionCount,
    causalChain,
    summary,
  }
}
