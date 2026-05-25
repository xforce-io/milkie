// Boundary engines for the context region substrate.
// Spec: docs/superpowers/specs/2026-05-25-context-region-substrate-design.md §7
//
// Two engines fire at distinct boundaries:
//   - runIntraTurnEngine: at each FSM step boundary (applies pending mutations,
//     handles state-scoped expiry, tool-buffer/one-shot decrement, TTL expiry)
//   - runInterTurnEngine: at turn-end (crystallization — archive final answer
//     into history pair region; drop turn-local; promote-to-wm)

import type { ContextRegions } from './ContextRegions'

interface ScratchAssistantContent {
  role:       'assistant'
  text:       string
  hasToolUse: boolean
}

export function extractFinalAssistantText(regions: ContextRegions): string {
  const candidates = [...regions._allRegions()]
    .filter(r => r.section === 'scratchpad')
    .filter(r => (r.content as { role?: string }).role === 'assistant')
    .sort((a, b) => {
      // Primary sort: by createdAt (descending — latest first)
      if (b.createdAt !== a.createdAt) return b.createdAt - a.createdAt
      // Secondary sort: by ordinal (descending — highest ordinal first)
      const aOrdinal = a.ordinal ?? 0
      const bOrdinal = b.ordinal ?? 0
      return bOrdinal - aOrdinal
    })
  for (const r of candidates) {
    const c = r.content as ScratchAssistantContent
    if (!c.hasToolUse && c.text) return c.text
  }
  return ''
}
