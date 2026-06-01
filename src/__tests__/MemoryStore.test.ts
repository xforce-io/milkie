import { MemoryStore } from '../store/MemoryStore'

describe('MemoryStore', () => {
  let store: MemoryStore

  beforeEach(() => {
    store = new MemoryStore()
  })

  it('set and get', async () => {
    await store.set('k', { x: 1 })
    expect(await store.get('k')).toEqual({ x: 1 })
  })

  it('returns undefined for missing key', async () => {
    expect(await store.get('missing')).toBeUndefined()
  })

  it('exists', async () => {
    expect(await store.exists('k')).toBe(false)
    await store.set('k', 'v')
    expect(await store.exists('k')).toBe(true)
  })

  it('delete', async () => {
    await store.set('k', 'v')
    await store.delete('k')
    expect(await store.get('k')).toBeUndefined()
  })

  it('TTL expiry', async () => {
    await store.set('k', 'v', -1)  // already expired
    expect(await store.get('k')).toBeUndefined()
  })

  it('list returns entries matching prefix, ignoring others', async () => {
    await store.set('context:c1:var:a', 1)
    await store.set('context:c1:var:b', 'two')
    await store.set('context:c2:var:x', 9)   // different context
    await store.set('unrelated', true)
    const got = await store.list('context:c1:var:')
    const obj = Object.fromEntries(got.map(e => [e.key, e.value]))
    expect(obj).toEqual({ 'context:c1:var:a': 1, 'context:c1:var:b': 'two' })
  })

  it('list skips expired entries', async () => {
    await store.set('p:live', 1)
    await store.set('p:dead', 2, -1)  // already expired
    const got = await store.list('p:')
    expect(got.map(e => e.key)).toEqual(['p:live'])
  })
})
