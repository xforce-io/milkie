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
  /** PR-D Phase 1: 'system-end' when any active system region declared cacheBreakpoint=true. */
  cacheBreakpoint?: 'system-end'
}

export function assemble(regions: ContextRegions, scope: AssembleScope): AssembledContext {
  const active = [...regions._allRegions()].filter(r => isActive(r, scope))

  const systemRegions  = active.filter(r => r.target === 'system')
  const messageRegions = active.filter(r => r.target === 'message')
  const toolRegions    = active.filter(r => r.target === 'tool')

  const systemBlocks: string[] = []
  for (const sec of SECTION_SCHEMA.system) {
    for (const r of systemRegions.filter(x => x.section === sec).sort(bySectionLocalOrder)) {
      systemBlocks.push(r.format(r.content) as string)
    }
  }

  const messages: Message[] = []
  for (const sec of SECTION_SCHEMA.message) {
    for (const r of messageRegions.filter(x => x.section === sec).sort(bySectionLocalOrder)) {
      const out = r.format(r.content)
      if (Array.isArray(out)) messages.push(...(out as Message[]))
      else messages.push(out as Message)
    }
  }

  const tools: ToolSchema[] = []
  for (const r of toolRegions.slice().sort(bySectionLocalOrder)) {
    tools.push(r.format(r.content) as ToolSchema)
  }

  const hasSystemBreakpoint = active.some(r => r.target === 'system' && r.cacheBreakpoint === true)

  return {
    system:   systemBlocks.join('\n'),
    messages,
    ...(tools.length > 0 ? { tools } : {}),
    ...(hasSystemBreakpoint ? { cacheBreakpoint: 'system-end' as const } : {}),
  }
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
