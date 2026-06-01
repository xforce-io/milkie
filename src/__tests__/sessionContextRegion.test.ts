import { assemble } from '../context/assemble'
import type { AssembleScope } from '../context/assemble'
import { ContextRegions } from '../context/ContextRegions'
import { makeSessionContextRegion, makeTurnContextRegion } from '../context/lifecycleEngine'
import type { RegionInput } from '../context/Region'
import type { Message } from '../types/common.js'

function scope(): AssembleScope {
  return { currentState: 'default', currentTurnId: 'turn-1', currentEpoch: 0 }
}

function msgRegion(section: 'history' | 'current-turn', text: string): RegionInput {
  return {
    target:    'message',
    section,
    intraTurn: 'turn-persistent',
    interTurn: section === 'history' ? 'session-persistent' : 'turn-local',
    stability: section === 'history' ? 'session-stable' : 'volatile',
    content:   { role: 'user' as const, content: [{ type: 'text' as const, text }] },
    format:    (c) => c as Message,
  }
}

function textOf(m: Message | undefined): string {
  const c = m?.content[0]
  return c && c.type === 'text' ? c.text : ''
}

describe('makeSessionContextRegion (#83)', () => {
  it('renders session vars into a message between history and turn-context', () => {
    const store = new ContextRegions(() => 0)
    store.set('history', msgRegion('history', 'PREV'))
    store.set('session-context', makeSessionContextRegion({ workspace_instructions: '用中文', session_id: 's-9' }))
    store.set('turn-context', makeTurnContextRegion({ current_time: '10:00' }))
    store.set('current-turn', msgRegion('current-turn', 'NOW'))

    const out = assemble(store, scope())

    // ordering: history → session-context → turn-context → current-turn
    expect(out.messages).toHaveLength(4)
    expect(textOf(out.messages[0])).toBe('PREV')
    expect(textOf(out.messages[3])).toBe('NOW')

    const session = textOf(out.messages[1])
    expect(session).toContain('Session Context')
    expect(session).toContain('workspace_instructions')
    expect(session).toContain('用中文')
    expect(session).toContain('session_id')
    expect(session).toContain('s-9')

    // turn-context sits after session-context
    expect(textOf(out.messages[2])).toContain('current_time')
  })

  it('never touches the system block (history-cache safety)', () => {
    const store = new ContextRegions(() => 0)
    store.set('header', {
      target: 'system', section: 'header', intraTurn: 'turn-persistent',
      interTurn: 'session-persistent', stability: 'immutable',
      content: 'You are an agent.', format: (c) => String(c),
    })
    store.set('session-context', makeSessionContextRegion({ workspace_instructions: '用中文' }))

    const out = assemble(store, scope())
    expect(out.system).toBe('You are an agent.')
    expect(out.system).not.toContain('workspace_instructions')
    expect(out.system).not.toContain('用中文')
  })

  it('renders byte-identical output for identical vars (sorted keys)', () => {
    const a = makeSessionContextRegion({ b: '2', a: '1' })
    const b = makeSessionContextRegion({ a: '1', b: '2' })
    expect(textOf(a.format(a.content) as Message)).toBe(textOf(b.format(b.content) as Message))
  })
})
