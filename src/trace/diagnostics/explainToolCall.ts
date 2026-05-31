import type { Event, EventKind } from '../types.js'
import { walkCausedBy } from './walkCausedBy.js'
import { summarizeEvent } from './summarizeEvent.js'

export interface ToolCallExplanation {
  toolRequestedEventId: string
  toolName: string
  input: unknown
  output?: unknown
  trigger: { causedByEventId?: string; causedBySummary?: string }
  causalChain: Array<{ eventId: string; type: EventKind; summary: string }>
  summary: string
}

/**
 * Pure projection: explain a tool call — its name/input, the paired output,
 * which llm.responded decided to call it, and the causal chain. No LLM/I/O.
 * Region/CLI #36 reuse the serializable result. Parallel to explainLlmCall.
 */
export function explainToolCall(events: Event[], toolRequestedEventId: string): ToolCallExplanation {
  const byId = new Map<string, Event>()
  for (const e of events) byId.set(e.id, e)

  const evt = byId.get(toolRequestedEventId)
  if (!evt) throw new Error(`explainToolCall: unknown event id "${toolRequestedEventId}"`)
  if (evt.type !== 'tool.requested') {
    throw new Error(`explainToolCall: event "${toolRequestedEventId}" is "${evt.type}", expected "tool.requested"`)
  }

  const p = evt.payload as { toolName?: unknown; input?: unknown; requestHash?: unknown }
  const toolName = String(p.toolName ?? '?')
  const responded = events.find(e =>
    e.type === 'tool.responded' && (e.payload as { requestHash?: unknown }).requestHash === p.requestHash)
  const output = responded ? (responded.payload as { output?: unknown }).output : undefined

  const causeEvt = evt.causedBy ? byId.get(evt.causedBy) : undefined
  const causeSummary = causeEvt ? summarizeEvent(causeEvt) : undefined

  const causalChain = walkCausedBy(events, toolRequestedEventId).map(e => ({
    eventId: e.id, type: e.type, summary: summarizeEvent(e),
  }))

  const summary = `工具 ${toolName} 被 ${causeSummary ?? '(无上游记录)'} 调用`

  return {
    toolRequestedEventId,
    toolName,
    input: p.input,
    ...(output !== undefined ? { output } : {}),
    trigger: {
      ...(evt.causedBy ? { causedByEventId: evt.causedBy } : {}),
      ...(causeSummary ? { causedBySummary: causeSummary } : {}),
    },
    causalChain,
    summary,
  }
}
