import type { Event, EventKind, LlmRespondedPayload, ToolRespondedPayload } from '../types.js'
import { walkCausedBy } from './walkCausedBy.js'
import { summarizeEvent } from './summarizeEvent.js'

/**
 * explainDecision — the #175 diagnostic anchor that replaces the old
 * `explainTransition` (which read `fsm.transition` business-topology nodes).
 *
 * A "decision" is one autonomous loop-body decision, anchored on a real I/O
 * EFFECT that already lives in the event log:
 *
 *   - `llm.responded`  — the model chose what to do: which tools to call (and
 *                        with what input) and/or what text to emit.
 *   - `tool.responded` — a tool produced a result the loop folds back in.
 *
 * Because the anchor is an effect (not authored control flow), this covers
 * every agent — single-state autonomous, slot-filling, sub-agent — and is
 * deterministic/replay-clean (effects are content-addressed in the log). The
 * `causedBy` chain is the causal spine (#30): who decided this call.
 */

export type DecisionAnchorKind = 'llm' | 'tool'

export interface DecisionExplanation {
  decisionEventId: string
  kind: DecisionAnchorKind
  /**
   * For an llm decision: the tools the model selected this turn (name + short
   * input) plus whether it emitted final text. For a tool decision: the tool
   * name and its ok/error outcome.
   */
  chose: {
    tools:    Array<{ name: string; input?: unknown }>
    text?:    string
    toolName?: string
    status?:  'ok' | 'error'
  }
  trigger: {
    causedByEventId?: string
    causedBySummary?: string
  }
  causalChain: Array<{ eventId: string; type: EventKind; summary: string }>
  summary: string
}

/**
 * Pure event-log projection: explain one decision (an `llm.responded` or
 * `tool.responded`). Reads only `events` — no LLM, no I/O, no stored snapshot.
 */
export function explainDecision(events: Event[], decisionEventId: string): DecisionExplanation {
  const byId = new Map<string, Event>()
  for (const e of events) byId.set(e.id, e)

  const evt = byId.get(decisionEventId)
  if (!evt) throw new Error(`explainDecision: unknown event id "${decisionEventId}"`)
  if (evt.type !== 'llm.responded' && evt.type !== 'tool.responded') {
    throw new Error(`explainDecision: event "${decisionEventId}" is "${evt.type}", expected "llm.responded" or "tool.responded"`)
  }

  const causeEvt = evt.causedBy ? byId.get(evt.causedBy) : undefined
  const causeSummary = causeEvt ? summarizeEvent(causeEvt) : undefined

  const causalChain = walkCausedBy(events, decisionEventId).map(e => ({
    eventId: e.id,
    type:    e.type,
    summary: summarizeEvent(e),
  }))

  const triggerSource = causeSummary ?? '(无上游记录)'

  if (evt.type === 'llm.responded') {
    const resp = (evt.payload as LlmRespondedPayload).response
    const tools = (resp?.toolCalls ?? []).map(c => ({ name: c.name, ...(c.input !== undefined ? { input: c.input } : {}) }))
    const textBlock = resp?.content?.find(
      (c): c is { type: 'text'; text: string } => c.type === 'text' && !!c.text,
    )
    const text = textBlock?.text
    const what = tools.length
      ? `调用 ${tools.map(t => t.name).join('、')}`
      : text ? `输出文本(完成)` : '无动作'
    const summary = `LLM 决策:${what};由 ${triggerSource} 触发`
    return {
      decisionEventId,
      kind: 'llm',
      chose: { tools, ...(text ? { text } : {}) },
      trigger: {
        ...(evt.causedBy ? { causedByEventId: evt.causedBy } : {}),
        ...(causeSummary ? { causedBySummary: causeSummary } : {}),
      },
      causalChain,
      summary,
    }
  }

  // tool.responded
  const p = evt.payload as ToolRespondedPayload
  const toolName = String(p.toolName ?? '?')
  const status: 'ok' | 'error' = p.status ?? (p.error ? 'error' : 'ok')
  const summary = `工具结果 ${toolName} (${status});由 ${triggerSource} 触发`
  return {
    decisionEventId,
    kind: 'tool',
    chose: { tools: [], toolName, status },
    trigger: {
      ...(evt.causedBy ? { causedByEventId: evt.causedBy } : {}),
      ...(causeSummary ? { causedBySummary: causeSummary } : {}),
    },
    causalChain,
    summary,
  }
}
