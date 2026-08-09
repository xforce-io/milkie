import type { Event } from '../types.js'
import { walkCausedBy } from './walkCausedBy.js'
import { decodeLlmOutcome } from '../LlmOutcome.js'
import { TraceIntegrityError } from '../TraceIntegrityError.js'

/**
 * #175 de-core: the decision spine no longer carries `fsm.transition` business
 * nodes. A decision is an I/O EFFECT — `llm.responded` (the model chose tools /
 * text, or failed) and `tool.responded` (a tool produced a result) — plus the final
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
  /** LLM branch only: success vs recorded failure. */
  status?:          'ok' | 'error'
  error?:           { code: string; message: string; phase: string }
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

function labelOf(e: Event): { label: string; status?: 'ok' | 'error'; error?: DecisionNode['error'] } {
  if (e.type === 'llm.responded') {
    try {
      const outcome = decodeLlmOutcome(e)
      if (outcome.status === 'error') {
        return {
          label: `LLM failure · ${outcome.error.code}`,
          status: 'error',
          error: {
            code: outcome.error.code,
            message: outcome.error.message,
            phase: outcome.error.phase,
          },
        }
      }
      const tools = outcome.response.toolCalls ?? []
      return {
        label: tools.length ? `LLM → ${tools.map(t => t.name).join(', ')}` : 'LLM → 文本',
        status: 'ok',
      }
    } catch (err) {
      if (err instanceof TraceIntegrityError) {
        return { label: `LLM integrity · ${err.kind}`, status: 'error' }
      }
      throw err
    }
  }
  if (e.type === 'tool.responded') {
    const payload = e.payload
    const toolName =
      payload && typeof payload === 'object' && 'toolName' in payload && typeof payload.toolName === 'string'
        ? payload.toolName
        : '?'
    return { label: `tool: ${toolName}` }
  }
  return { label: '输出' }
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
      const labeled = labelOf(e)
      return {
        eventId:   e.id,
        kind:      kindOf(e.type),
        label:     labeled.label,
        timestamp: e.timestamp,
        ...(labeled.status ? { status: labeled.status } : {}),
        ...(labeled.error ? { error: labeled.error } : {}),
        ...(e.causedBy ? { causedByEventId: e.causedBy } : {}),
        ...(ancestor ? { causeDecisionId: ancestor.id } : {}),
      }
    })
    .sort((a, b) => a.timestamp - b.timestamp)
  return { nodes }
}
