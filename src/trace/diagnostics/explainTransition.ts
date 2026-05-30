import type { Event, EventKind, FsmEventDomain, FsmTransitionPayload, GuardEvaluation } from '../types.js'
import { walkCausedBy } from './walkCausedBy.js'
import { summarizeEvent } from './summarizeEvent.js'

export interface TransitionExplanation {
  transitionEventId: string
  from: string
  to:   string
  trigger: {
    name:             string
    domain:           FsmEventDomain
    causedByEventId?: string
    causedBySummary?: string
  }
  guards: GuardEvaluation[]
  causalChain: Array<{ eventId: string; type: EventKind; summary: string }>
  summary: string
}

/**
 * Pure event-log projection: explain why a given fsm.transition fired.
 * Reads only `events` — no LLM, no I/O, no stored snapshot. Returns a plain
 * serializable object (also the JSON shape the CLI explainer #36 will emit).
 */
export function explainTransition(events: Event[], transitionEventId: string): TransitionExplanation {
  const byId = new Map<string, Event>()
  for (const e of events) byId.set(e.id, e)

  const evt = byId.get(transitionEventId)
  if (!evt) throw new Error(`explainTransition: unknown event id "${transitionEventId}"`)
  if (evt.type !== 'fsm.transition') {
    throw new Error(`explainTransition: event "${transitionEventId}" is "${evt.type}", expected "fsm.transition"`)
  }

  const p = evt.payload as FsmTransitionPayload
  const causeEvt = evt.causedBy ? byId.get(evt.causedBy) : undefined
  const guards = p.guardEvaluations ?? []

  const causalChain = walkCausedBy(events, transitionEventId).map(e => ({
    eventId: e.id,
    type:    e.type,
    summary: summarizeEvent(e),
  }))

  const guardPart = guards.length
    ? `;guard ${guards.map(g => `${g.guardId} 判定 ${String(g.result)}`).join('、')}`
    : ''
  const triggerSource = causeEvt ? summarizeEvent(causeEvt) : '(无上游记录)'
  const summary = `${p.from} → ${p.to}:由 ${triggerSource} 发出的 ${p.trigger.name} 触发${guardPart}`

  return {
    transitionEventId,
    from: p.from,
    to:   p.to,
    trigger: {
      name:   p.trigger.name,
      domain: p.trigger.domain,
      ...(evt.causedBy ? { causedByEventId: evt.causedBy } : {}),
      ...(causeEvt ? { causedBySummary: summarizeEvent(causeEvt) } : {}),
    },
    guards,
    causalChain,
    summary,
  }
}
