// Boundary engines for the context region substrate.
// Spec: docs/superpowers/specs/2026-05-25-context-region-substrate-design.md §7
//
// Two engines fire at distinct boundaries:
//   - runIntraTurnEngine: at each FSM step boundary (applies pending mutations,
//     handles state-scoped expiry, tool-buffer/one-shot decrement, TTL expiry)
//   - runInterTurnEngine: at turn-end (crystallization — archive final answer
//     into history pair region; drop turn-local; promote-to-wm)

import type { ContextRegions } from './ContextRegions'
import type { RegionInput } from './Region'
import type { Message, MessageContent } from '../types/common.js'
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
