// Pure assembly function: regions Map → assembled ModelRequest parts.
// Spec: docs/superpowers/specs/2026-05-25-context-region-substrate-design.md §5
//
// Called once per LLM request boundary by AgentRuntime (PR-C). No mutation,
// no IOPort access — deterministic by construction: same regions + same scope
// → byte-identical output.

import type { Region } from './Region'
import type { ContextRegions } from './ContextRegions'
import { SECTION_SCHEMA } from './sectionSchema'
import type { Message } from '../types/common.js'
import type { ToolSchema } from '../types/model.js'
import { messagesFromBudgetItems, type ContextBudgetItem, type ContextBudgetRegion } from './budget.js'
import {
  CURRENT_USER_MESSAGE_MARKER,
  currentTurnInput,
  renderDeliveredContextBlock,
} from './lifecycleEngine.js'
import type { ContextProjection } from '../types/common.js'

export interface AssembleScope {
  currentState:  string
  currentTurnId: string
  currentEpoch:  number
  subAgentId?:   string
}

// The assembled parts that come from regions. The caller (AgentRuntime)
// composes a full ModelRequest by adding model / toolChoice / metadata
// from agent config — those concerns don't belong in regions.
export interface AssembledContext {
  system:   string
  messages: Message[]
  tools?:   ToolSchema[]
  /** Source-preserving items for the request-boundary budget planner. */
  budgetItems: ContextBudgetItem[]
  /** PR-D Phase 1: 'system-end' when any active system region declared cacheBreakpoint=true. */
  cacheBreakpoint?: 'system-end'
}

export function assemble(regions: ContextRegions, scope: AssembleScope): AssembledContext {
  const active = [...regions._allRegions()].filter(r => isActive(r, scope))

  const systemRegions  = active.filter(r => r.target === 'system')
  const messageRegions = active.filter(r => r.target === 'message')
  const toolRegions    = active.filter(r => r.target === 'tool')

  const budgetItems: ContextBudgetItem[] = []
  let order = 0
  for (const sec of SECTION_SCHEMA.system) {
    for (const r of systemRegions.filter(x => x.section === sec).sort(bySectionLocalOrder)) {
      budgetItems.push({
        id:      r.id,
        region:  budgetRegionFor(r),
        target:  'system',
        content: r.format(r.content) as string,
        order:   order++,
        evidenceRef: evidenceRefFor(r),
        rawContent: r.content,
      })
    }
  }

  for (const sec of SECTION_SCHEMA.message) {
    for (const r of messageRegions.filter(x => x.section === sec).sort(bySectionLocalOrder)) {
      if (r.id === 'current-turn' && isProjectedCurrentTurn(r.content)) {
        const liveInput = currentTurnInput(r.content)
        const prefix = `${CURRENT_USER_MESSAGE_MARKER}\n`
        for (const [index, projection] of r.content.projections.entries()) {
          budgetItems.push({
            id:      `external-projection:${projection.sourceRunId}:${index}`,
            region:  'externalProjection',
            target:  'message',
            content: [{
              role:    'user',
              content: [{ type: 'text', text: renderDeliveredContextBlock([projection]) }],
            }],
            order:       order++,
            mergeKey:    'current-turn',
            evidenceRef: `run:${projection.sourceRunId}`,
            rawContent:  projection,
          })
        }
        budgetItems.push({
          id:      r.id,
          region:  'currentTurn',
          target:  'message',
          content: [{
            role:    'user',
            content: [{ type: 'text', text: `${prefix}${liveInput}` }],
          }],
          order:       order++,
          mergeKey:    'current-turn',
          mergePrefix: prefix,
          evidenceRef: `turn:${r.id}`,
        })
        continue
      }
      if (isVariableRegion(r)) {
        const title = r.section === 'session-context' ? '--- Session Context ---' : '--- Turn Context ---'
        for (const key of Object.keys(r.content).sort()) {
          const value = r.content[key]
          budgetItems.push({
            id:      `${r.id}:${key}`,
            region:  'sessionContext',
            target:  'message',
            content: [{ role: 'user', content: [{ type: 'text', text: `${title}\n${key}: ${formatVariable(value)}` }] }],
            order:       order++,
            mergeKey:    r.id,
            evidenceRef: `region:${r.id}:${key}`,
            rawContent:  value,
          })
        }
        continue
      }
      const out = r.format(r.content)
      budgetItems.push({
        id:      r.id,
        region:  budgetRegionFor(r),
        target:  'message',
        content: Array.isArray(out) ? out as Message[] : [out as Message],
        order:   order++,
        evidenceRef: evidenceRefFor(r),
        rawContent: r.content,
      })
    }
  }

  for (const r of toolRegions.slice().sort(bySectionLocalOrder)) {
    budgetItems.push({
      id:      r.id,
      region:  'control',
      target:  'tool',
      content: [r.format(r.content) as ToolSchema],
      order:   order++,
      evidenceRef: evidenceRefFor(r),
      rawContent: r.content,
    })
  }

  const hasSystemBreakpoint = active.some(r => r.target === 'system' && r.cacheBreakpoint === true)

  const system = budgetItems
    .filter((item): item is ContextBudgetItem & { content: string } => item.target === 'system')
    .map(item => item.content)
    .join('\n')
  const messages = messagesFromBudgetItems(budgetItems)
  const tools = budgetItems
    .filter((item): item is ContextBudgetItem & { content: ToolSchema[] } => item.target === 'tool')
    .flatMap(item => item.content)

  return {
    system,
    messages,
    ...(tools.length > 0 ? { tools } : {}),
    budgetItems,
    ...(hasSystemBreakpoint ? { cacheBreakpoint: 'system-end' as const } : {}),
  }
}

function isProjectedCurrentTurn(content: unknown): content is { input: string; projections: ContextProjection[] } {
  return !!content
    && typeof content === 'object'
    && Array.isArray((content as { projections?: unknown }).projections)
    && (content as { projections: ContextProjection[] }).projections.length > 0
}

function isVariableRegion(region: Region): region is Region & { content: Record<string, unknown> } {
  return (region.section === 'session-context' || region.section === 'turn-context')
    && !!region.content
    && typeof region.content === 'object'
    && !Array.isArray(region.content)
}

function formatVariable(value: unknown): string {
  return typeof value === 'string' ? value : JSON.stringify(value)
}

function budgetRegionFor(region: Region): ContextBudgetRegion {
  if (region.target === 'tool') return 'control'
  if (region.target === 'system') return region.section === 'wm' ? 'workingMemory' : 'control'
  if (region.section === 'history') return 'history'
  if (region.section === 'external-context') return 'externalProjection'
  if (region.section === 'session-context' || region.section === 'turn-context') return 'sessionContext'
  if (region.section === 'current-turn') return 'currentTurn'
  return 'scratchpad'
}

function evidenceRefFor(region: Region): string {
  if (region.section === 'scratchpad') {
    const raw = (region.content as { raw?: Array<{ type?: string; id?: string; tool_use_id?: string }> }).raw ?? []
    const toolPart = raw.find(part => part.type === 'tool_result' || part.type === 'tool_use')
    const toolUseId = toolPart?.tool_use_id ?? toolPart?.id
    if (toolUseId) return `tool-call:${toolUseId}`
  }
  if (region.section === 'wm') return `working-memory:${region.id}`
  return `region:${region.id}`
}

// Per spec §5: only compare ordinal when BOTH regions declared one; otherwise
// fall back to createdAt. Partial ordinal usage is intentionally meaningless
// — agents either commit to ordinals for a section or rely on createdAt.
function bySectionLocalOrder(a: Region, b: Region): number {
  if (a.ordinal != null && b.ordinal != null) return a.ordinal - b.ordinal
  return a.createdAt - b.createdAt
}

// Per spec §5: assemble filters regions that are not active in the current
// scope. Belt-and-suspenders alongside the boundary engines (PR-C) which
// proactively delete such regions — assemble still hides them defensively so
// a missed engine pass cannot leak stale content into the LLM request.
//
// Other lifecycle states (one-shot, tool-buffer, turn-local, summarize,
// promote-to-wm) are engine-driven mutations, not runtime filters, so they
// are not checked here.
function isActive(region: Region, scope: AssembleScope): boolean {
  if (typeof region.intraTurn === 'object' && region.intraTurn.kind === 'state-scoped') {
    if (region.intraTurn.state !== scope.currentState) return false
  }
  if (typeof region.interTurn === 'object' && region.interTurn.kind === 'ttl') {
    if (scope.currentEpoch > region.interTurn.deadline) return false
  }
  return true
}
