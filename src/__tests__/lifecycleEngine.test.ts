import { ContextRegions } from '../context/ContextRegions'
import { extractFinalAssistantText } from '../context/lifecycleEngine'
import type { RegionInput } from '../context/Region'

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
