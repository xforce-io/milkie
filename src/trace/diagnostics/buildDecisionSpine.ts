import type { Event } from '../types.js'
import { walkCausedBy } from './walkCausedBy.js'

export type DecisionKind = 'llm' | 'tool' | 'transition' | 'output'

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

const SPINE_TYPES = new Set(['llm.requested', 'tool.requested', 'fsm.transition', 'agent.run.completed'])

function kindOf(type: string): DecisionKind {
  if (type === 'llm.requested')  return 'llm'
  if (type === 'tool.requested') return 'tool'
  if (type === 'fsm.transition') return 'transition'
  return 'output'
}

function labelOf(e: Event): string {
  if (e.type === 'llm.requested')  return 'LLM call'
  if (e.type === 'tool.requested') return `tool: ${String((e.payload as { toolName?: unknown }).toolName ?? '?')}`
  if (e.type === 'fsm.transition') { const p = e.payload as { from: string; to: string }; return `${p.from} → ${p.to}` }
  return '输出'
}

/**
 * Project the event log down to the decision spine: only LLM calls, tool
 * calls, transitions, and the final output, in timestamp order. For each node,
 * causeDecisionId is the nearest decision ancestor reached by walking causedBy
 * (skipping non-decision causes like tool.responded / run.started). Pure.
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
