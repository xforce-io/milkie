import type { Event, LlmRequestedPayload, ToolRequestedPayload, ToolRespondedPayload } from '../types.js'
import { contextRefsAt, type RegionContentRef } from '../RegionContextView.js'
import {
  decodeLlmOutcome,
  failureViewOf,
  type LlmFailureView,
  type LlmOutcome,
} from '../LlmOutcome.js'
import { TraceIntegrityError } from '../TraceIntegrityError.js'

export interface CacheHealth {
  tier:             'hot' | 'warm' | 'cold'
  readTokens:       number
  creationTokens:   number
  totalInputTokens: number
  hitRate:          number
}

export interface RegionGroup {
  stability: string
  regions:   RegionContentRef[]
}

export interface ToolStep {
  name:    string
  input?:  unknown
  output?: unknown
  error?:  unknown
  status:  'ok' | 'error' | 'pending'
}

export interface ExecutionStep {
  kind:          'llm' | 'tool'
  label:         string
  /** LLM step lifecycle: pending (no terminal yet), ok, or error. */
  status?:       'pending' | 'ok' | 'error'
  messageCount?: number
  cacheHealth?:  CacheHealth | null
  regionGroups?: RegionGroup[]
  prompt?:       { system?: unknown; messages: unknown[]; tools: unknown[] } | null
  /** Present only on status:'ok'. */
  response?:     unknown
  /** Present only on status:'error'. */
  error?:        LlmFailureView
  tool?:         ToolStep
}

export interface ExecutionProjection {
  steps: ExecutionStep[]
}

// Canonical prompt-assembly order: most stable on top, volatile scratch last.
// Lifted from the frontend's STABILITY_ORDER so the projection owns the rule.
const STABILITY_ORDER = ['immutable', 'session-stable', 'turn-stable', 'volatile'] as const

function groupRegionsByStability(refs: RegionContentRef[]): RegionGroup[] {
  const groups: RegionGroup[] = []
  for (const stability of STABILITY_ORDER) {
    const regions = refs.filter(r => r.stability === stability)
    if (regions.length > 0) groups.push({ stability, regions })
  }
  return groups
}

/**
 * Cache-health tiering — lifted verbatim from the agent-docs-qa frontend's
 * classifyCacheTier so the projection owns the rule, not the UI.
 *   hot  — read-dominated (hit ≥ 0.7)
 *   warm — some reuse OR a fresh cache entry was written (substrate engaged)
 *   cold — cache ran but neither read nor wrote
 */
function classifyCacheTier(c: { hitRate?: number; creationTokens?: number }): CacheHealth['tier'] {
  const hit = c.hitRate ?? 0
  const created = c.creationTokens ?? 0
  if (hit >= 0.7) return 'hot'
  if (hit >= 0.3 || created > 0) return 'warm'
  return 'cold'
}

/**
 * Pure event-log projection of a run's execution timeline: one step per
 * llm.requested / tool.requested, carrying the cache-health tier and (later)
 * region composition the frontend used to recompute itself. No I/O.
 *
 * LLM terminals are paired by causedBy (not hash overwrite) so concurrent same-hash
 * invocations and failure terminals stay distinct.
 */
export function buildExecutionProjection(
  events: Event[],
  opts: { regionContent?: Map<string, string> } = {},
): ExecutionProjection {
  const regionContent = opts.regionContent
  // Pair terminals by causedBy → requestEventId (precise).
  const llmTerminalByRequestId = new Map<string, { event: Event; outcome: LlmOutcome }>()
  const toolResponses = new Map<string, Event>()

  for (const e of events) {
    if (e.type === 'llm.responded') {
      let outcome: LlmOutcome
      try {
        outcome = decodeLlmOutcome(e)
      } catch (err) {
        if (err instanceof TraceIntegrityError) continue
        throw err
      }
      if (typeof e.causedBy === 'string' && e.causedBy.length > 0) {
        llmTerminalByRequestId.set(e.causedBy, { event: e, outcome })
      }
    } else if (e.type === 'tool.responded') {
      const p = e.payload as ToolRespondedPayload
      if (p.requestHash) toolResponses.set(p.requestHash, e)
    }
  }

  const steps: ExecutionStep[] = []
  for (const e of events) {
    if (e.type === 'llm.requested') {
      const p = e.payload as LlmRequestedPayload
      const paired = llmTerminalByRequestId.get(e.id)
      const outcome = paired?.outcome
      const cacheStats = outcome?.status === 'ok' ? outcome.cacheStats : undefined
      const refs = Array.from(contextRefsAt(events, e.id, 'at').values()).map(r =>
        r.contentHash && regionContent?.has(r.contentHash)
          ? { ...r, content: regionContent.get(r.contentHash) }
          : r,
      )
      const req = (p.request ?? {}) as { system?: unknown; messages?: unknown[]; tools?: unknown[] }
      const messages = Array.isArray(req.messages) ? req.messages : []
      const status: ExecutionStep['status'] = !outcome
        ? 'pending'
        : outcome.status === 'ok' ? 'ok' : 'error'
      const label = status === 'error' && outcome && outcome.status === 'error'
        ? `LLM failure · ${outcome.error.code}`
        : 'LLM call'
      steps.push({
        kind:         'llm',
        label,
        status,
        messageCount: messages.length,
        cacheHealth:  cacheStats
          ? { tier: classifyCacheTier(cacheStats), ...cacheStats }
          : null,
        regionGroups: groupRegionsByStability(refs),
        prompt:       { ...(req.system !== undefined ? { system: req.system } : {}), messages, tools: Array.isArray(req.tools) ? req.tools : [] },
        ...(outcome?.status === 'ok' ? { response: outcome.response } : {}),
        ...(outcome?.status === 'error' ? { error: failureViewOf(outcome.error) } : {}),
      })
    } else if (e.type === 'tool.requested') {
      const p = e.payload as ToolRequestedPayload
      const resp = p.requestHash ? toolResponses.get(p.requestHash) : undefined
      const respPayload = resp?.payload as ToolRespondedPayload | undefined
      const status: ToolStep['status'] = !respPayload
        ? 'pending'
        : respPayload.error !== undefined ? 'error' : 'ok'
      steps.push({
        kind:  'tool',
        label: `Tool · ${p.toolName}`,
        tool:  {
          name:   p.toolName,
          input:  p.input,
          ...(respPayload?.output !== undefined ? { output: respPayload.output } : {}),
          ...(respPayload?.error  !== undefined ? { error:  respPayload.error  } : {}),
          status,
        },
      })
    }
  }
  return { steps }
}
