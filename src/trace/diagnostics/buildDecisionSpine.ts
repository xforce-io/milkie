import type { Event } from '../types.js'
import { walkCausedBy } from './walkCausedBy.js'

/**
 * #175 de-core: the decision spine no longer carries `fsm.transition` business
 * nodes. A decision is an I/O EFFECT — `llm.responded` (the model chose tools /
 * text) and `tool.responded` (a tool produced a result) — plus the final
 * output. This is the same anchor `explainDecision` reads.
 */
export type DecisionKind = 'llm' | 'tool' | 'output'

export interface DecisionNode {
  eventId:          string
  kind:             DecisionKind
  label:            string
  timestamp:        number
  causedByEventId?: string
  causeDecisionId?: string
}

export interface DecisionSpine {
  nodes: DecisionNode[]
}

const SPINE_TYPES = new Set(['llm.responded', 'tool.responded', 'agent.run.completed'])

function kindOf(type: string): DecisionKind {
  if (type === 'llm.responded')  return 'llm'
  if (type === 'tool.responded') return 'tool'
  return 'output'
}

function labelOf(e: Event): string {
  if (e.type === 'llm.responded') {
    const resp = (e.payload as { response?: { toolCalls?: Array<{ name: string }> } }).response
    const tools = resp?.toolCalls ?? []
    return tools.length ? `LLM → ${tools.map(t => t.name).join(', ')}` : 'LLM → 文本'
  }
  if (e.type === 'tool.responded') return `tool: ${String((e.payload as { toolName?: unknown }).toolName ?? '?')}`
  return '输出'
}

/**
 * Project the event log down to the decision spine: only llm decisions, tool
 * results, and the final output, in timestamp order. For each node,
 * causeDecisionId is the nearest decision ancestor reached by walking causedBy
 * (skipping non-decision causes). Pure.
 */
export function buildDecisionSpine(events: Event[]): DecisionSpine {
  const spineIds = new Set(events.filter(e => SPINE_TYPES.has(e.type)).map(e => e.id))
  const nodes: DecisionNode[] = events
    .filter(e => SPINE_TYPES.has(e.type))
    .map(e => {
      const ancestor = walkCausedBy(events, e.id).slice(1).find(a => spineIds.has(a.id))
      return {
        eventId:   e.id,
        kind:      kindOf(e.type),
        label:     labelOf(e),
        timestamp: e.timestamp,
        ...(e.causedBy ? { causedByEventId: e.causedBy } : {}),
        ...(ancestor ? { causeDecisionId: ancestor.id } : {}),
      }
    })
    .sort((a, b) => a.timestamp - b.timestamp)
  return { nodes }
}
