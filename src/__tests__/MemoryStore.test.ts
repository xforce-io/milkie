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
})
