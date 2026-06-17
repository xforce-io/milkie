import type { Event, EventKind } from '../types.js'
import { walkCausedBy } from './walkCausedBy.js'
import { summarizeEvent } from './summarizeEvent.js'
import { contextRefsAt } from '../RegionContextView.js'

export interface LlmCallExplanation {
  llmRequestedEventId: string
  trigger: {
    causedByEventId?: string
    causedBySummary?: string
  }
  regionCount: number
  causalChain: Array<{ eventId: string; type: EventKind; summary: string }>
  summary: string
}

/**
 * Pure event-log projection: explain why a given llm.requested fired — the
 * turn-terminator that triggered it, how many regions composed the prompt, and
 * the causal chain. No LLM, no I/O, no stored snapshot.
 *
 * #175 de-core: the old `fsmState` field is gone — there is no longer an
 * authored business-state machine to report a state name from. The causal chain
 * (causedBy) already explains what led here.
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
  const regionCount = contextRefsAt(events, llmRequestedEventId, 'at').size

  const causalChain = walkCausedBy(events, llmRequestedEventId).map(e => ({
    eventId: e.id,
    type:    e.type,
    summary: summarizeEvent(e),
  }))

  const triggerSource = causeSummary ?? '(无上游记录)'
  const summary = `LLM 调用,由 ${triggerSource} 触发;prompt 由 ${regionCount} 个 region 拼成`

  return {
    llmRequestedEventId,
    trigger: {
      ...(evt.causedBy ? { causedByEventId: evt.causedBy } : {}),
      ...(causeSummary ? { causedBySummary: causeSummary } : {}),
    },
    regionCount,
    causalChain,
    summary,
  }
}
