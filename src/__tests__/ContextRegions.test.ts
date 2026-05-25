import { ContextRegions } from '../context/ContextRegions'
import type { RegionInput } from '../context/Region'

// Tiny test fixture builder; tests fill in only the fields they care about.
function regionInput(overrides: Partial<RegionInput> = {}): RegionInput {
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

describe('ContextRegions — set / get / delete primitives', () => {
  test('get on empty store returns undefined', () => {
    const store = new ContextRegions(() => 0)
    expect(store.get('missing')).toBeUndefined()
  })

  test('set then get returns the region with id and createdAt filled', () => {
    const store = new ContextRegions(() => 42)
    store.set('header:base', regionInput({ content: 'base prompt' }))
    const r = store.get('header:base')
    expect(r).toBeDefined()
    expect(r!.id).toBe('header:base')
    expect(r!.createdAt).toBe(42)
    expect(r!.content).toBe('base prompt')
  })

  test('delete returns true when region existed', () => {
    const store = new ContextRegions(() => 0)
    store.set('x', regionInput())
    expect(store.delete('x')).toBe(true)
  })

  test('delete returns false when region did not exist', () => {
    const store = new ContextRegions(() => 0)
    expect(store.delete('nonexistent')).toBe(false)
  })

  test('get after delete returns undefined', () => {
    const store = new ContextRegions(() => 0)
    store.set('x', regionInput())
    store.delete('x')
    expect(store.get('x')).toBeUndefined()
  })
})

describe('ContextRegions — upsert preserves createdAt + epoch tracking', () => {
  test('set on existing id keeps the original createdAt (sort-stable upsert)', () => {
    let now = 10
    const store = new ContextRegions(() => now)
    store.set('x', regionInput({ content: 'v1' }))
    now = 999
    store.set('x', regionInput({ content: 'v2' }))
    const r = store.get('x')
    expect(r!.createdAt).toBe(10)
    expect(r!.content).toBe('v2')
  })

  test('epoch starts at 0', () => {
    const store = new ContextRegions(() => 0)
    expect(store.getEpoch()).toBe(0)
  })

  test('set increments epoch', () => {
    const store = new ContextRegions(() => 0)
    store.set('x', regionInput())
    expect(store.getEpoch()).toBe(1)
    store.set('y', regionInput())
    expect(store.getEpoch()).toBe(2)
  })

  test('upsert (set on existing id) also increments epoch', () => {
    const store = new ContextRegions(() => 0)
    store.set('x', regionInput())
    expect(store.getEpoch()).toBe(1)
    store.set('x', regionInput({ content: 'v2' }))
    expect(store.getEpoch()).toBe(2)
  })

  test('delete of existing region increments epoch', () => {
    const store = new ContextRegions(() => 0)
    store.set('x', regionInput())
    const before = store.getEpoch()
    store.delete('x')
    expect(store.getEpoch()).toBe(before + 1)
  })

  test('delete of non-existent region does NOT increment epoch', () => {
    const store = new ContextRegions(() => 0)
    store.set('x', regionInput())
    const before = store.getEpoch()
    store.delete('nonexistent')
    expect(store.getEpoch()).toBe(before)
  })
})

describe('ContextRegions — clock injection', () => {
  test('set calls clock exactly once per new region', () => {
    const clock = jest.fn(() => 7)
    const store = new ContextRegions(clock)
    store.set('a', regionInput())
    store.set('b', regionInput())
    expect(clock).toHaveBeenCalledTimes(2)
  })

  test('upsert (set on existing id) does NOT call clock — createdAt preserved', () => {
    const clock = jest.fn(() => 7)
    const store = new ContextRegions(clock)
    store.set('a', regionInput())
    clock.mockClear()
    store.set('a', regionInput({ content: 'updated' }))
    expect(clock).not.toHaveBeenCalled()
  })

  test('get does not call clock', () => {
    const clock = jest.fn(() => 7)
    const store = new ContextRegions(clock)
    store.set('a', regionInput())
    clock.mockClear()
    store.get('a')
    store.get('missing')
    expect(clock).not.toHaveBeenCalled()
  })

  test('delete does not call clock', () => {
    const clock = jest.fn(() => 7)
    const store = new ContextRegions(clock)
    store.set('a', regionInput())
    clock.mockClear()
    store.delete('a')
    store.delete('missing')
    expect(clock).not.toHaveBeenCalled()
  })

  test('sequential clock values are stamped in order — sort-stable createdAt', () => {
    const values = [100, 200, 300]
    let i = 0
    const store = new ContextRegions(() => values[i++]!)
    store.set('a', regionInput())
    store.set('b', regionInput())
    store.set('c', regionInput())
    expect(store.get('a')!.createdAt).toBe(100)
    expect(store.get('b')!.createdAt).toBe(200)
    expect(store.get('c')!.createdAt).toBe(300)
  })
})

describe('ContextRegions — snapshot / restore', () => {
  test('snapshot of empty store captures epoch=0 and no regions', () => {
    const store = new ContextRegions(() => 0)
    const snap = store.snapshot()
    expect(snap.epoch).toBe(0)
    expect(snap.regions).toEqual([])
  })

  test('snapshot captures current epoch and all regions', () => {
    const store = new ContextRegions(() => 50)
    store.set('a', regionInput({ content: 'one' }))
    store.set('b', regionInput({ content: 'two' }))
    const snap = store.snapshot()
    expect(snap.epoch).toBe(2)
    expect(snap.regions).toHaveLength(2)
    const ids = snap.regions.map(r => r.id).sort()
    expect(ids).toEqual(['a', 'b'])
  })

  test('restore on fresh store reproduces regions and epoch', () => {
    const src = new ContextRegions(() => 11)
    src.set('a', regionInput({ content: 'one' }))
    src.set('b', regionInput({ content: 'two' }))
    src.delete('a')
    const snap = src.snapshot()

    const dst = new ContextRegions(() => 999)   // different clock — should not be called
    dst.restore(snap)
    expect(dst.getEpoch()).toBe(snap.epoch)
    expect(dst.get('a')).toBeUndefined()
    expect(dst.get('b')!.content).toBe('two')
    expect(dst.get('b')!.createdAt).toBe(11)    // original createdAt preserved
  })

  test('restore replaces any existing regions (not a merge)', () => {
    const dst = new ContextRegions(() => 0)
    dst.set('old', regionInput({ content: 'should-be-cleared' }))

    const src = new ContextRegions(() => 5)
    src.set('new', regionInput({ content: 'survives' }))
    dst.restore(src.snapshot())

    expect(dst.get('old')).toBeUndefined()
    expect(dst.get('new')!.content).toBe('survives')
  })

  test('snapshot is decoupled from source store (later mutation does not leak)', () => {
    const store = new ContextRegions(() => 0)
    store.set('a', regionInput())
    const snap = store.snapshot()
    store.set('b', regionInput())
    expect(snap.regions).toHaveLength(1)
    expect(snap.epoch).toBe(1)
  })

  test('restore does not invoke clock', () => {
    const src = new ContextRegions(() => 5)
    src.set('a', regionInput())
    const snap = src.snapshot()

    const clock = jest.fn(() => 999)
    const dst = new ContextRegions(clock)
    dst.restore(snap)
    expect(clock).not.toHaveBeenCalled()
  })
})

describe('ContextRegions — _allRegions iteration', () => {
  test('empty store iterates zero times', () => {
    const store = new ContextRegions(() => 0)
    expect([..._spread(store)]).toEqual([])
  })

  test('iterates all current regions (count + ids)', () => {
    const store = new ContextRegions(() => 0)
    store.set('a', regionInput({ content: '1' }))
    store.set('b', regionInput({ content: '2' }))
    store.set('c', regionInput({ content: '3' }))
    const ids = [..._spread(store)].map(r => r.id).sort()
    expect(ids).toEqual(['a', 'b', 'c'])
  })

  test('does not include deleted regions', () => {
    const store = new ContextRegions(() => 0)
    store.set('a', regionInput())
    store.set('b', regionInput())
    store.delete('a')
    const ids = [..._spread(store)].map(r => r.id)
    expect(ids).toEqual(['b'])
  })
})

function _spread(store: ContextRegions): IterableIterator<import('../context/Region').Region> {
  return store._allRegions()
}
