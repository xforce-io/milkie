import { assemble } from '../context/assemble'
import type { AssembleScope } from '../context/assemble'
import { ContextRegions } from '../context/ContextRegions'
import { makeTurnContextRegion } from '../context/lifecycleEngine'
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

describe('makeTurnContextRegion (#82)', () => {
  it('renders variables into a message placed between history and current-turn', () => {
    const store = new ContextRegions(() => 0)
    store.set('history', msgRegion('history', 'PREV'))
    store.set('turn-context', makeTurnContextRegion({ current_time: 'T1', workspace: 'W' }))
    store.set('current-turn', msgRegion('current-turn', 'NOW'))

    const out = assemble(store, scope())

    // ordering: history → turn-context → current-turn
    expect(out.messages).toHaveLength(3)
    expect(textOf(out.messages[0])).toBe('PREV')
    expect(textOf(out.messages[2])).toBe('NOW')

    // the middle message carries the injected variables (key + value readable)
    const mid = textOf(out.messages[1])
    expect(mid).toContain('current_time')
    expect(mid).toContain('T1')
    expect(mid).toContain('workspace')
    expect(mid).toContain('W')
  })

  it('never touches the system block (prefix-cache safety)', () => {
    const store = new ContextRegions(() => 0)
    store.set('header', {
      target: 'system', section: 'header', intraTurn: 'turn-persistent',
      interTurn: 'session-persistent', stability: 'immutable',
      content: 'You are an agent.', format: (c) => String(c),
    })
    store.set('turn-context', makeTurnContextRegion({ secret_var: 'XYZ' }))

    const out = assemble(store, scope())
    expect(out.system).toBe('You are an agent.')
    expect(out.system).not.toContain('secret_var')
    expect(out.system).not.toContain('XYZ')
  })

  it('renders byte-identical output for identical variables (sorted keys)', () => {
    const a = makeTurnContextRegion({ b: '2', a: '1' })
    const b = makeTurnContextRegion({ a: '1', b: '2' })
    expect(textOf(a.format(a.content) as Message)).toBe(textOf(b.format(b.content) as Message))
  })
})
