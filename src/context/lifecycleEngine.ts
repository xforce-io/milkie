// Boundary engines for the context region substrate.
// Spec: docs/superpowers/specs/2026-05-25-context-region-substrate-design.md §7
//
// Two engines fire at distinct boundaries:
//   - runIntraTurnEngine: at each FSM step boundary (applies pending mutations,
//     handles state-scoped expiry, tool-buffer/one-shot decrement, TTL expiry)
//   - runInterTurnEngine: at turn-end (crystallization — archive final answer
//     into history pair region; drop turn-local; promote-to-wm)

import type { ContextRegions } from './ContextRegions'
import type { Region, RegionInput, RegionSnapshot } from './Region'
import type { Message, MessageContent, JSONValue } from '../types/common.js'
import type { ToolSchema } from '../types/model.js'

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

export function makeHeaderRegion(systemPrompt: string): RegionInput {
  return {
    target:    'system',
    section:   'header',
    intraTurn: 'turn-persistent',
    interTurn: 'session-persistent',
    stability: 'immutable',
    content:   systemPrompt,
    format:    (c) => String(c),
  }
}

export function makeSkillRegion(
  name: string,
  instructions: string,
  scope: 'turn' | 'session' = 'session',
): RegionInput {
  return {
    target:    'system',
    section:   scope === 'session' ? 'persistent-skills' : 'session-skills',
    intraTurn: 'turn-persistent',
    interTurn: scope === 'session' ? 'session-persistent' : 'turn-local',
    stability: scope === 'session' ? 'session-stable' : 'turn-stable',
    content:   { name, instructions },
    format:    (c) => {
      const { name, instructions } = c as { name: string; instructions: string }
      return `\n--- Skill: ${name} ---\n${instructions}`
    },
  }
}

export function makeStateInstructionsRegion(
  state: string,
  instructions: string,
): RegionInput {
  return {
    target:    'system',
    section:   'state',
    intraTurn: { kind: 'state-scoped', state },
    interTurn: 'turn-local',
    stability: 'turn-stable',
    content:   instructions,
    format:    (c) => `\n--- Current Instructions ---\n${String(c)}`,
  }
}

export function makeWmRegion(
  data: Record<string, unknown>,
  log: unknown[],
): RegionInput | null {
  if (Object.keys(data).length === 0 && log.length === 0) return null
  // Sort keys for deterministic byte-identical output across runs.
  const sortedData: Record<string, unknown> = {}
  for (const k of Object.keys(data).sort()) sortedData[k] = data[k]
  return {
    target:    'system',
    section:   'wm',
    intraTurn: 'turn-persistent',
    interTurn: 'session-persistent',
    stability: 'volatile',
    content:   { data: sortedData, log },
    format:    (c) =>
      '\n--- Working Memory ---\n' + JSON.stringify(c, null, 2),
  }
}

// #82/#83: shared compact `key: value` renderer. Sorted keys → byte-identical output
// for the same vars (deterministic, replay-friendly); non-string values JSON-encoded.
function formatVarsBlock(title: string, content: unknown): Message {
  const vars = content as Record<string, JSONValue>
  const lines = Object.keys(vars).sort().map((k) => {
    const v = vars[k]
    return `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`
  })
  return {
    role:    'user',
    content: [{ type: 'text', text: `${title}\n${lines.join('\n')}` }],
  }
}

export function makeSessionContextRegion(variables: Record<string, JSONValue>): RegionInput {
  return {
    target:    'message',
    section:   'session-context',
    intraTurn: 'turn-persistent',
    interTurn: 'session-persistent',
    stability: 'session-stable',
    content:   variables,
    format:    (c): Message => formatVarsBlock('--- Session Context ---', c),
  }
}

export function makeTurnContextRegion(variables: Record<string, JSONValue>): RegionInput {
  return {
    target:    'message',
    section:   'turn-context',
    intraTurn: 'turn-persistent',
    interTurn: 'turn-local',
    stability: 'volatile',
    content:   variables,
    format:    (c): Message => formatVarsBlock('--- Turn Context ---', c),
  }
}

export function makeCurrentTurnRegion(input: string): RegionInput {
  return {
    target:    'message',
    section:   'current-turn',
    intraTurn: 'turn-persistent',
    interTurn: 'turn-local',
    stability: 'volatile',
    content:   input,
    format:    (c): Message => ({
      role:    'user',
      content: [{ type: 'text', text: String(c) }],
    }),
  }
}

export function makeScratchpadAssistantRegion(
  content: MessageContent[],
  hasToolUse: boolean,
): RegionInput {
  // hasToolUse stored on the region's content metadata so extractFinalAssistantText
  // can skip intermediate (tool-bearing) assistant turns without parsing the message.
  // The text field is the concatenation of any text parts (used by crystallization).
  const text = content
    .filter(c => c.type === 'text')
    .map(c => (c as { text: string }).text)
    .join('')
  return {
    target:    'message',
    section:   'scratchpad',
    intraTurn: 'turn-persistent',
    interTurn: 'turn-local',
    stability: 'volatile',
    content:   { role: 'assistant' as const, text, hasToolUse, raw: content },
    format:    (c): Message => ({
      role:    'assistant',
      content: (c as { raw: MessageContent[] }).raw,
    }),
  }
}

export function makeScratchpadToolResultRegion(
  content: MessageContent[],
): RegionInput {
  return {
    target:    'message',
    section:   'scratchpad',
    intraTurn: 'turn-persistent',
    interTurn: 'turn-local',
    stability: 'volatile',
    content:   { role: 'tool' as const, raw: content },
    format:    (c): Message => ({
      role:    'tool',
      content: (c as { raw: MessageContent[] }).raw,
    }),
  }
}

export function makeHistoryPairRegion(
  userInput: string,
  assistantText: string,
): RegionInput {
  return {
    target:    'message',
    section:   'history',
    intraTurn: 'turn-persistent',
    interTurn: 'session-persistent',
    stability: 'session-stable',
    content:   { userInput, assistantText },
    format:    (c): Message[] => {
      const { userInput, assistantText } = c as { userInput: string; assistantText: string }
      return [
        { role: 'user',      content: [{ type: 'text', text: userInput }] },
        { role: 'assistant', content: [{ type: 'text', text: assistantText }] },
      ]
    },
  }
}

export function makeToolSchemaRegion(schema: ToolSchema): RegionInput {
  return {
    target:    'tool',
    section:   'default',
    intraTurn: 'turn-persistent',
    interTurn: 'session-persistent',
    stability: 'session-stable',
    content:   schema,
    format:    (c) => c as ToolSchema,
  }
}

export interface InterTurnContext {
  boundary:   'turn-end' | 'turn-start'
  userInput?: string
  now:        number
  /** PR-D: fires once after crystallization completes (turn-end only) with the summary. */
  onBoundary?: (summary: CrystallizationSummary) => void
}

export interface CrystallizationSummary {
  kept:         string[]
  dropped:      string[]
  promoted:     Array<{ from: string; to: string }>
  archivedPair: string | undefined
}

export function runInterTurnEngine(
  regions: ContextRegions,
  ctx: InterTurnContext,
): { crystallization?: CrystallizationSummary } {
  if (ctx.boundary !== 'turn-end') return {}

  const summary: CrystallizationSummary = {
    kept: [], dropped: [], promoted: [], archivedPair: undefined,
  }

  // Step 1: archive final answer (if userInput available)
  if (ctx.userInput !== undefined) {
    const finalText = extractFinalAssistantText(regions)
    const pairId = `history:turn-${ctx.now}`
    regions.set(pairId, makeHistoryPairRegion(ctx.userInput, finalText))
    summary.archivedPair = pairId
  }

  // Step 2: iterate snapshot of all regions, apply per-interTurn rule.
  // Use a snapshot (`[...regions._allRegions()]`) because we mutate during iteration.
  for (const r of [...regions._allRegions()]) {
    if (r.id === summary.archivedPair) {
      summary.kept.push(r.id)
      continue
    }
    if (r.interTurn === 'session-persistent') {
      summary.kept.push(r.id)
      continue
    }
    if (r.interTurn === 'turn-local') {
      regions.delete(r.id)
      summary.dropped.push(r.id)
      continue
    }
    if (typeof r.interTurn === 'object' && r.interTurn.kind === 'ttl') {
      if (ctx.now > r.interTurn.deadline) {
        regions.delete(r.id)
        summary.dropped.push(r.id)
      } else {
        summary.kept.push(r.id)
      }
      continue
    }
    if (r.interTurn === 'promote-to-wm') {
      const promotedId = `wm:${r.id}`
      regions.set(promotedId, {
        target:    'system',
        section:   'wm',
        intraTurn: 'turn-persistent',
        interTurn: 'session-persistent',
        stability: 'session-stable',
        content:   r.content,
        format:    r.format,
      })
      regions.delete(r.id)
      summary.promoted.push({ from: r.id, to: promotedId })
      continue
    }
    if (r.interTurn === 'summarize-on-overflow') {
      // Phase 1: treat as session-persistent; budget summarization is future work.
      summary.kept.push(r.id)
      continue
    }
  }

  ctx.onBoundary?.(summary)
  return { crystallization: summary }
}

// Re-attach format functions to regions loaded from a serialized snapshot
// (e.g. checkpoint stored as JSON in SQLite). JSON round-trip drops functions;
// dispatch by id prefix or section to pick the right factory's format.
//
// Spec: docs/superpowers/specs/2026-05-25-context-region-substrate-design.md §4.2
//   (snapshot/restore) — restore consumers re-attach format. Region.format is
//   transient runtime state, not durable data.
export function rehydrateRegion(r: Region): Region {
  if (typeof r.format === 'function') return r   // already has it

  // Choose factory by id prefix / section. Each branch reconstructs the same
  // format closure the corresponding factory produced.
  if (r.id === 'header') {
    return { ...r, format: (c: unknown) => String(c) }
  }
  if (r.id.startsWith('skill:')) {
    return {
      ...r,
      format: (c: unknown) => {
        const { name, instructions } = c as { name: string; instructions: string }
        return `\n--- Skill: ${name} ---\n${instructions}`
      },
    }
  }
  if (r.id.startsWith('state-instr:')) {
    return { ...r, format: (c: unknown) => `\n--- Current Instructions ---\n${String(c)}` }
  }
  if (r.id === 'wm' || r.id.startsWith('wm:')) {
    return { ...r, format: (c: unknown) => '\n--- Working Memory ---\n' + JSON.stringify(c, null, 2) }
  }
  if (r.id === 'current-turn') {
    return {
      ...r,
      format: (c: unknown): Message => ({
        role:    'user',
        content: [{ type: 'text', text: String(c) }],
      }),
    }
  }
  if (r.id.startsWith('scratch:')) {
    const role = (r.content as { role?: string }).role
    if (role === 'assistant') {
      return {
        ...r,
        format: (c: unknown): Message => ({
          role:    'assistant',
          content: (c as { raw: MessageContent[] }).raw,
        }),
      }
    }
    return {
      ...r,
      format: (c: unknown): Message => ({
        role:    'tool',
        content: (c as { raw: MessageContent[] }).raw,
      }),
    }
  }
  if (r.id.startsWith('history:turn-')) {
    return {
      ...r,
      format: (c: unknown): Message[] => {
        const { userInput, assistantText } = c as { userInput: string; assistantText: string }
        return [
          { role: 'user',      content: [{ type: 'text', text: userInput }] },
          { role: 'assistant', content: [{ type: 'text', text: assistantText }] },
        ]
      },
    }
  }
  if (r.id.startsWith('tool:')) {
    return { ...r, format: (c: unknown) => c as ToolSchema }
  }
  // Unknown id pattern: identity format so assemble doesn't crash. The region
  // likely won't render anything useful, but at least won't throw.
  return { ...r, format: (c: unknown) => String(c) }
}

export function rehydrateSnapshot(snap: RegionSnapshot): RegionSnapshot {
  return {
    epoch:   snap.epoch,
    regions: snap.regions.map(rehydrateRegion),
  }
}
