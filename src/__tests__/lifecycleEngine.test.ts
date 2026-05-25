import { ContextRegions } from '../context/ContextRegions'
import {
  extractFinalAssistantText,
  makeScratchpadAssistantRegion,
  makeScratchpadToolResultRegion,
  makeCurrentTurnRegion,
  makeHistoryPairRegion,
  makeHeaderRegion,
  makeSkillRegion,
  makeStateInstructionsRegion,
  makeWmRegion,
  makeToolSchemaRegion,
} from '../context/lifecycleEngine'
import type { RegionInput } from '../context/Region'
import type { MessageContent } from '../types/common.js'
import type { ToolSchema } from '../types/model.js'

function scratchAssistantRegion(text: string, ordinal: number, hasToolUse = false): RegionInput {
  return {
    target:    'message',
    section:   'scratchpad',
    intraTurn: 'turn-persistent',
    interTurn: 'turn-local',
    stability: 'volatile',
    ordinal,
    content:   { role: 'assistant', text, hasToolUse },
    format:    () => ({ role: 'assistant', content: [] }),
  }
}

describe('extractFinalAssistantText', () => {
  test('returns empty string when no scratchpad regions', () => {
    const r = new ContextRegions(() => 0)
    expect(extractFinalAssistantText(r)).toBe('')
  })

  test('returns the latest assistant region without tool_use', () => {
    const r = new ContextRegions(() => 0)
    r.set('s1', scratchAssistantRegion('first answer', 1, false))
    r.set('s2', scratchAssistantRegion('second answer', 2, false))
    expect(extractFinalAssistantText(r)).toBe('second answer')
  })

  test('skips assistant regions that contain tool_use', () => {
    const r = new ContextRegions(() => 0)
    r.set('s1', scratchAssistantRegion('thinking...', 1, true))
    r.set('s2', scratchAssistantRegion('final answer', 2, false))
    expect(extractFinalAssistantText(r)).toBe('final answer')
  })

  test('ignores non-scratchpad regions', () => {
    const r = new ContextRegions(() => 0)
    r.set('hist', {
      target: 'message', section: 'history',
      intraTurn: 'turn-persistent', interTurn: 'session-persistent', stability: 'session-stable',
      content: { role: 'assistant', text: 'old', hasToolUse: false },
      format: () => ({ role: 'assistant', content: [] }),
    })
    expect(extractFinalAssistantText(r)).toBe('')
  })
})

describe('region factories', () => {
  test('makeHeaderRegion: target=system, section=header, immutable, session-persistent', () => {
    const r = makeHeaderRegion('You are an agent.')
    expect(r.target).toBe('system')
    expect(r.section).toBe('header')
    expect(r.stability).toBe('immutable')
    expect(r.interTurn).toBe('session-persistent')
    expect(r.format(r.content)).toBe('You are an agent.')
  })

  test('makeSkillRegion(turn): target=system, section=session-skills, interTurn=turn-local', () => {
    const r = makeSkillRegion('verifier', 'INSTRUCTIONS', 'turn')
    expect(r.section).toBe('session-skills')
    expect(r.interTurn).toBe('turn-local')
    expect(r.format(r.content)).toContain('verifier')
    expect(r.format(r.content)).toContain('INSTRUCTIONS')
  })

  test('makeSkillRegion(session): section=persistent-skills, interTurn=session-persistent', () => {
    const r = makeSkillRegion('helper', 'INST', 'session')
    expect(r.section).toBe('persistent-skills')
    expect(r.interTurn).toBe('session-persistent')
  })

  test('makeStateInstructionsRegion: state-scoped intraTurn, section=state', () => {
    const r = makeStateInstructionsRegion('researching', 'Focus on the chapter content.')
    expect(r.section).toBe('state')
    expect(r.intraTurn).toEqual({ kind: 'state-scoped', state: 'researching' })
    expect(r.format(r.content)).toContain('Focus on the chapter content.')
  })

  test('makeWmRegion: section=wm, deterministic key order in JSON', () => {
    const r1 = makeWmRegion({ b: 2, a: 1, c: 3 }, [])
    const r2 = makeWmRegion({ a: 1, b: 2, c: 3 }, [])
    // Same data, different insertion order → same serialized output (sorted keys)
    expect(r1!.format(r1!.content)).toBe(r2!.format(r2!.content))
  })

  test('makeWmRegion: omitted when data + log are both empty', () => {
    const r = makeWmRegion({}, [])
    expect(r).toBeNull()
  })

  test('makeCurrentTurnRegion: target=message, section=current-turn, turn-local', () => {
    const r = makeCurrentTurnRegion('hello')
    expect(r.target).toBe('message')
    expect(r.section).toBe('current-turn')
    expect(r.interTurn).toBe('turn-local')
    const msg = r.format(r.content) as { role: string; content: MessageContent[] }
    expect(msg.role).toBe('user')
    expect((msg.content[0] as { text: string }).text).toBe('hello')
  })

  test('makeScratchpadAssistantRegion: turn-local, section=scratchpad, role=assistant', () => {
    const content: MessageContent[] = [{ type: 'text', text: 'thinking' }]
    const r = makeScratchpadAssistantRegion(content, false)
    expect(r.section).toBe('scratchpad')
    expect(r.interTurn).toBe('turn-local')
    const msg = r.format(r.content) as { role: string; content: MessageContent[] }
    expect(msg.role).toBe('assistant')
    expect(msg.content).toEqual(content)
  })

  test('makeScratchpadToolResultRegion: tool message with tool_result content', () => {
    const content: MessageContent[] = [
      { type: 'tool_result', tool_use_id: 'tc1', content: 'ok' },
    ]
    const r = makeScratchpadToolResultRegion(content)
    expect(r.section).toBe('scratchpad')
    expect(r.interTurn).toBe('turn-local')
    const msg = r.format(r.content) as { role: string; content: MessageContent[] }
    expect(msg.role).toBe('tool')
  })

  test('makeHistoryPairRegion: returns Message[] (user + assistant)', () => {
    const r = makeHistoryPairRegion('what time?', 'noon')
    expect(r.section).toBe('history')
    expect(r.interTurn).toBe('session-persistent')
    const msgs = r.format(r.content) as { role: string; content: MessageContent[] }[]
    expect(msgs).toHaveLength(2)
    expect(msgs[0]!.role).toBe('user')
    expect((msgs[0]!.content[0] as { text: string }).text).toBe('what time?')
    expect(msgs[1]!.role).toBe('assistant')
    expect((msgs[1]!.content[0] as { text: string }).text).toBe('noon')
  })

  test('makeToolSchemaRegion: target=tool, section=default', () => {
    const schema: ToolSchema = { name: 'echo', description: 'e', inputSchema: {} }
    const r = makeToolSchemaRegion(schema)
    expect(r.target).toBe('tool')
    expect(r.section).toBe('default')
    expect(r.format(r.content)).toBe(schema)
  })
})
