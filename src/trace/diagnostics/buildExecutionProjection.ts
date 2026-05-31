import type { Event, LlmRequestedPayload, LlmRespondedPayload, ToolRequestedPayload, ToolRespondedPayload } from '../types.js'
import { contextRefsAt, type RegionContentRef } from '../RegionContextView.js'

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
  messageCount?: number
  cacheHealth?:  CacheHealth | null
  regionGroups?: RegionGroup[]
  prompt?:       { system?: unknown; messages: unknown[]; tools: unknown[] } | null
  response?:     unknown
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
 */
export function buildExecutionProjection(
  events: Event[],
  opts: { regionContent?: Map<string, string> } = {},
): ExecutionProjection {
  const regionContent = opts.regionContent
  const llmResponses = new Map<string, Event>()
  const toolResponses = new Map<string, Event>()
  for (const e of events) {
    if (e.type === 'llm.responded') {
      const p = e.payload as LlmRespondedPayload
      if (p.requestHash) llmResponses.set(p.requestHash, e)
    } else if (e.type === 'tool.responded') {
      const p = e.payload as ToolRespondedPayload
      if (p.requestHash) toolResponses.set(p.requestHash, e)
    }
  }

  const steps: ExecutionStep[] = []
  for (const e of events) {
    if (e.type === 'llm.requested') {
      const p = e.payload as LlmRequestedPayload
      const resp = p.requestHash ? llmResponses.get(p.requestHash) : undefined
      const cacheStats = resp && (resp.payload as LlmRespondedPayload).cacheStats
      const refs = Array.from(contextRefsAt(events, e.id, 'at').values()).map(r =>
        r.contentHash && regionContent?.has(r.contentHash)
          ? { ...r, content: regionContent.get(r.contentHash) }
          : r,
      )
      const req = (p.request ?? {}) as { system?: unknown; messages?: unknown[]; tools?: unknown[] }
      const messages = Array.isArray(req.messages) ? req.messages : []
      steps.push({
        kind:         'llm',
        label:        'LLM call',
        messageCount: messages.length,
        cacheHealth:  cacheStats
          ? { tier: classifyCacheTier(cacheStats), ...cacheStats }
          : null,
        regionGroups: groupRegionsByStability(refs),
        prompt:       { ...(req.system !== undefined ? { system: req.system } : {}), messages, tools: Array.isArray(req.tools) ? req.tools : [] },
        response:     (resp?.payload as LlmRespondedPayload | undefined)?.response,
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
