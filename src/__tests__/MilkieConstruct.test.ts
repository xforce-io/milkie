import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'

describe('Milkie construction (#99)', () => {
  it('throws when constructed without an explicit stateStore (no silent in-memory fallback)', () => {
    // `as any` simulates a JS caller / type-escape that omits stateStore.
    expect(() => new Milkie({} as any)).toThrow(/stateStore/)
  })

  it('throws when stateStore is explicitly undefined', () => {
    expect(() => new Milkie({ stateStore: undefined } as any)).toThrow(/stateStore/)
  })

  it('constructs fine with an explicit stateStore', () => {
    expect(() => new Milkie({ stateStore: new MemoryStore() })).not.toThrow()
  })
})
