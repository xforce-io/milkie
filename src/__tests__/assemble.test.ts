import { assemble } from '../context/assemble'
import type { AssembleScope } from '../context/assemble'
import { ContextRegions } from '../context/ContextRegions'
import type { RegionInput } from '../context/Region'
import type { Message } from '../types/common.js'
import type { ToolSchema } from '../types/model.js'

function defaultScope(overrides: Partial<AssembleScope> = {}): AssembleScope {
  return {
    currentState:  'default',
    currentTurnId: 'turn-1',
    currentEpoch:  0,
    ...overrides,
  }
}

function systemRegion(overrides: Partial<RegionInput> = {}): RegionInput {
  return {
    target:    'system',
    section:   'header',
    intraTurn: 'turn-persistent',
    interTurn: 'session-persistent',
    stability: 'immutable',
    content:   'hello',
    format:    (c) => String(c),
    ...overrides,
  }
}

function messageRegion(overrides: Partial<RegionInput> = {}): RegionInput {
  return {
    target:    'message',
    section:   'current-turn',
    intraTurn: 'turn-persistent',
    interTurn: 'turn-local',
    stability: 'volatile',
    content:   { role: 'user' as const, content: [{ type: 'text' as const, text: 'hi' }] },
    format:    (c) => c as Message,
    ...overrides,
  }
}

function toolRegion(overrides: Partial<RegionInput> = {}): RegionInput {
  return {
    target:    'tool',
    section:   'default',
    intraTurn: 'turn-persistent',
    interTurn: 'session-persistent',
    stability: 'session-stable',
    content:   { name: 'echo', description: 'echo', inputSchema: {} } as ToolSchema,
    format:    (c) => c as ToolSchema,
    ...overrides,
  }
}

describe('assemble — base cases', () => {
  test('empty regions → empty assembled context', () => {
    const store = new ContextRegions(() => 0)
    const out = assemble(store, defaultScope())
    expect(out.system).toBe('')
    expect(out.messages).toEqual([])
    expect(out.tools).toBeUndefined()
  })

  test('single system region → system block contains its formatted content', () => {
    const store = new ContextRegions(() => 0)
    store.set('header', systemRegion({ content: 'You are an agent.' }))
    const out = assemble(store, defaultScope())
    expect(out.system).toBe('You are an agent.')
    expect(out.messages).toEqual([])
    expect(out.tools).toBeUndefined()
  })

  test('single message region → messages array contains formatted Message', () => {
    const store = new ContextRegions(() => 0)
    store.set('u1', messageRegion())
    const out = assemble(store, defaultScope())
    expect(out.messages).toHaveLength(1)
    expect(out.messages[0]!.role).toBe('user')
  })

  test('single tool region → tools array contains schema', () => {
    const store = new ContextRegions(() => 0)
    store.set('t-echo', toolRegion())
    const out = assemble(store, defaultScope())
    expect(out.tools).toHaveLength(1)
    expect(out.tools![0]!.name).toBe('echo')
  })
})

describe('assemble — section ordering', () => {
  test('sections render in SECTION_SCHEMA order regardless of insertion order', () => {
    // Insert 'state' first, then 'header' — header must still come first.
    let now = 100
    const store = new ContextRegions(() => now++)
    store.set('state-1', systemRegion({ section: 'state',  content: 'STATE-INSTR' }))
    store.set('header',  systemRegion({ section: 'header', content: 'HEADER' }))
    const out = assemble(store, defaultScope())
    expect(out.system).toBe('HEADER\nSTATE-INSTR')
  })

  test('within section, regions sort by createdAt even when clock is non-monotonic', () => {
    // Force createdAt order to differ from Map insertion order.
    // Inserted A, B, C in that order; createdAt is 100, 50, 200.
    // Map.values() yields A, B, C; sorted by createdAt yields B, A, C.
    const values = [100, 50, 200]
    let i = 0
    const store = new ContextRegions(() => values[i++]!)
    store.set('a', systemRegion({ section: 'session-skills', content: 'A' }))
    store.set('b', systemRegion({ section: 'session-skills', content: 'B' }))
    store.set('c', systemRegion({ section: 'session-skills', content: 'C' }))
    const out = assemble(store, defaultScope())
    expect(out.system).toBe('B\nA\nC')
  })

  test('within section, when both regions have ordinal, ordinal beats createdAt', () => {
    let now = 0
    const store = new ContextRegions(() => now++)
    // createdAt order is X(0) then Y(1), but ordinals reverse them.
    store.set('x', systemRegion({ section: 'session-skills', content: 'X', ordinal: 20 }))
    store.set('y', systemRegion({ section: 'session-skills', content: 'Y', ordinal: 10 }))
    const out = assemble(store, defaultScope())
    expect(out.system).toBe('Y\nX')
  })

  test('within section, when only some have ordinal, fall back to createdAt for all (per spec §5)', () => {
    let now = 0
    const store = new ContextRegions(() => now++)
    store.set('p', systemRegion({ section: 'session-skills', content: 'P', ordinal: 99 }))  // createdAt=0
    store.set('q', systemRegion({ section: 'session-skills', content: 'Q' }))               // createdAt=1, no ordinal
    const out = assemble(store, defaultScope())
    // Per spec: bySectionLocalOrder only uses ordinal when BOTH have it.
    // P (createdAt=0) wins regardless of its high ordinal.
    expect(out.system).toBe('P\nQ')
  })

  test('messages render in SECTION_SCHEMA.message order: history → current-turn → scratchpad', () => {
    let now = 100
    const store = new ContextRegions(() => now++)
    // Insert in reverse order; expected order follows schema.
    store.set('scratch-1', messageRegion({
      section: 'scratchpad',
      content: { role: 'assistant' as const, content: [{ type: 'text' as const, text: 'thinking' }] },
    }))
    store.set('current', messageRegion({
      section: 'current-turn',
      content: { role: 'user' as const, content: [{ type: 'text' as const, text: 'now' }] },
    }))
    store.set('hist-1', messageRegion({
      section: 'history',
      content: { role: 'user' as const, content: [{ type: 'text' as const, text: 'past' }] },
    }))
    const out = assemble(store, defaultScope())
    expect(out.messages.map(m => (m.content[0] as { text: string }).text))
      .toEqual(['past', 'now', 'thinking'])
  })

  test('format returning Message[] is flattened into messages array (history pair case)', () => {
    const store = new ContextRegions(() => 0)
    const pair = {
      pair: [
        { role: 'user' as const,      content: [{ type: 'text' as const, text: 'Q' }] },
        { role: 'assistant' as const, content: [{ type: 'text' as const, text: 'A' }] },
      ],
    }
    store.set('hist-pair', {
      target:    'message',
      section:   'history',
      intraTurn: 'turn-persistent',
      interTurn: 'session-persistent',
      stability: 'session-stable',
      content:   pair,
      format:    (c) => (c as typeof pair).pair as Message[],
    })
    const out = assemble(store, defaultScope())
    expect(out.messages).toHaveLength(2)
    expect(out.messages[0]!.role).toBe('user')
    expect(out.messages[1]!.role).toBe('assistant')
  })

  test('multiple tool regions all included; sorted by createdAt', () => {
    const values = [200, 100]
    let i = 0
    const store = new ContextRegions(() => values[i++]!)
    store.set('t-b', toolRegion({
      content: { name: 'b', description: 'b', inputSchema: {} } as ToolSchema,
    }))
    store.set('t-a', toolRegion({
      content: { name: 'a', description: 'a', inputSchema: {} } as ToolSchema,
    }))
    const out = assemble(store, defaultScope())
    expect(out.tools!.map(t => t.name)).toEqual(['a', 'b'])  // a has createdAt=100, comes first
  })

  test('full SECTION_SCHEMA traversal: all system sections render in their declared order', () => {
    const store = new ContextRegions(() => 0)
    // Reverse insertion order vs schema, plus one per section.
    store.set('foot',  systemRegion({ section: 'footer',           content: 'FOOTER' }))
    store.set('wm',    systemRegion({ section: 'wm',               content: 'WM' }))
    store.set('toolss',systemRegion({ section: 'tools-state',      content: 'TOOLS-STATE' }))
    store.set('state', systemRegion({ section: 'state',            content: 'STATE' }))
    store.set('ssk',   systemRegion({ section: 'session-skills',   content: 'SESSION-SKILLS' }))
    store.set('tools0',systemRegion({ section: 'tools-static',     content: 'TOOLS-STATIC' }))
    store.set('psk',   systemRegion({ section: 'persistent-skills',content: 'PSK' }))
    store.set('hdr',   systemRegion({ section: 'header',           content: 'HDR' }))
    const out = assemble(store, defaultScope())
    expect(out.system).toBe(
      'HDR\nPSK\nTOOLS-STATIC\nSESSION-SKILLS\nSTATE\nTOOLS-STATE\nWM\nFOOTER'
    )
  })
})

describe('assemble — scope filtering (isActive)', () => {
  test('state-scoped region included only when scope.currentState matches', () => {
    const store = new ContextRegions(() => 0)
    store.set('s-a', systemRegion({
      section:   'state',
      content:   'A-INSTR',
      intraTurn: { kind: 'state-scoped', state: 'state-A' },
    }))
    store.set('s-b', systemRegion({
      section:   'state',
      content:   'B-INSTR',
      intraTurn: { kind: 'state-scoped', state: 'state-B' },
    }))
    const outA = assemble(store, defaultScope({ currentState: 'state-A' }))
    expect(outA.system).toBe('A-INSTR')
    const outB = assemble(store, defaultScope({ currentState: 'state-B' }))
    expect(outB.system).toBe('B-INSTR')
  })

  test('TTL region excluded when currentEpoch > deadline', () => {
    const store = new ContextRegions(() => 0)
    store.set('temp', systemRegion({
      content:   'EPHEMERAL',
      interTurn: { kind: 'ttl', deadline: 100 },
    }))
    const before = assemble(store, defaultScope({ currentEpoch: 50 }))
    expect(before.system).toBe('EPHEMERAL')
    const atDeadline = assemble(store, defaultScope({ currentEpoch: 100 }))
    expect(atDeadline.system).toBe('EPHEMERAL')   // boundary: deadline inclusive
    const after = assemble(store, defaultScope({ currentEpoch: 101 }))
    expect(after.system).toBe('')                 // excluded
  })

  test('turn-persistent / session-persistent always included regardless of scope', () => {
    const store = new ContextRegions(() => 0)
    store.set('a', systemRegion({
      content:   'ALWAYS',
      intraTurn: 'turn-persistent',
      interTurn: 'session-persistent',
    }))
    const out = assemble(store, defaultScope({
      currentState: 'whatever',
      currentEpoch: Number.MAX_SAFE_INTEGER,
    }))
    expect(out.system).toBe('ALWAYS')
  })

  test('state-scoped and TTL are AND-ed (region excluded if either fails)', () => {
    const store = new ContextRegions(() => 0)
    store.set('s', systemRegion({
      content:   'SCOPED-TTL',
      intraTurn: { kind: 'state-scoped', state: 'A' },
      interTurn: { kind: 'ttl', deadline: 100 },
    }))
    // wrong state
    expect(assemble(store, defaultScope({ currentState: 'B', currentEpoch: 0 })).system).toBe('')
    // right state, past deadline
    expect(assemble(store, defaultScope({ currentState: 'A', currentEpoch: 200 })).system).toBe('')
    // right state, within deadline
    expect(assemble(store, defaultScope({ currentState: 'A', currentEpoch: 50 })).system).toBe('SCOPED-TTL')
  })
})

describe('assemble — purity invariant', () => {
  test('same regions + scope → deep-equal AssembledContext on repeated calls', () => {
    const store = new ContextRegions(() => 0)
    store.set('h', systemRegion({ content: 'H' }))
    store.set('s', systemRegion({ section: 'session-skills', content: 'S' }))
    store.set('m', messageRegion())
    store.set('t', toolRegion())
    const scope = defaultScope({ currentState: 'x', currentEpoch: 10 })
    const a = assemble(store, scope)
    const b = assemble(store, scope)
    expect(b).toEqual(a)
  })

  test('assemble does not mutate the regions Map (epoch unchanged)', () => {
    const store = new ContextRegions(() => 0)
    store.set('a', systemRegion())
    store.set('b', messageRegion())
    const epochBefore = store.getEpoch()
    assemble(store, defaultScope())
    assemble(store, defaultScope())
    expect(store.getEpoch()).toBe(epochBefore)
  })

  test('assemble produces stable output across many invocations', () => {
    const store = new ContextRegions(() => 100)
    store.set('h',  systemRegion({ content: 'Base' }))
    store.set('s1', systemRegion({ section: 'session-skills', content: 'S1' }))
    store.set('s2', systemRegion({ section: 'session-skills', content: 'S2' }))
    const scope = defaultScope()
    const first = JSON.stringify(assemble(store, scope))
    for (let i = 0; i < 10; i++) {
      expect(JSON.stringify(assemble(store, scope))).toBe(first)
    }
  })
})

describe('assemble — cacheBreakpoint computation', () => {
  test('returns no cacheBreakpoint when no region marks one', () => {
    const store = new ContextRegions(() => 0)
    store.set('h', systemRegion())
    const out = assemble(store, defaultScope())
    expect(out.cacheBreakpoint).toBeUndefined()
  })

  test('returns "system-end" when any system region has cacheBreakpoint=true', () => {
    const store = new ContextRegions(() => 0)
    store.set('h', systemRegion())
    store.set('skill', systemRegion({
      section:        'persistent-skills',
      content:        'I',
      cacheBreakpoint: true,
    }))
    const out = assemble(store, defaultScope())
    expect(out.cacheBreakpoint).toBe('system-end')
  })

  test('cacheBreakpoint stays undefined if only message/tool regions mark it (Phase 1 only handles system-end)', () => {
    const store = new ContextRegions(() => 0)
    store.set('h', systemRegion())
    store.set('hist', messageRegion({ section: 'history', cacheBreakpoint: true }))
    const out = assemble(store, defaultScope())
    expect(out.cacheBreakpoint).toBeUndefined()
  })
})
