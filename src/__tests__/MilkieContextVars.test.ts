import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'

class DummyGateway implements IModelGateway {
  async complete(_r: ModelRequest): Promise<ModelResponse> {
    return { content: [], toolCalls: [], finishReason: 'end_turn' }
  }
  async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] }
}

const make = () => new Milkie({ stateStore: new MemoryStore(), gateway: new DummyGateway() })

describe('Milkie context vars (#83)', () => {
  it('set then get round-trips a value', async () => {
    const m = make()
    await m.setContextVar('c1', 'workspace_instructions', '用中文')
    expect(await m.getContextVar('c1', 'workspace_instructions')).toBe('用中文')
  })

  it('get returns undefined for a missing var', async () => {
    expect(await make().getContextVar('c1', 'nope')).toBeUndefined()
  })

  it('list returns all vars of a context keyed by name, isolated per context', async () => {
    const m = make()
    await m.setContextVar('c1', 'a', 1)
    await m.setContextVar('c1', 'b', 'two')
    await m.setContextVar('c2', 'x', 9)
    expect(await m.listContextVars('c1')).toEqual({ a: 1, b: 'two' })
    expect(await m.listContextVars('c2')).toEqual({ x: 9 })
  })

  it('delete removes a var', async () => {
    const m = make()
    await m.setContextVar('c1', 'a', 1)
    await m.deleteContextVar('c1', 'a')
    expect(await m.getContextVar('c1', 'a')).toBeUndefined()
  })

  it('var namespace does not collide with checkpoint/interrupt/children keys', async () => {
    const m = make()
    // a var literally named "interrupt" lives under context:c1:var:interrupt,
    // not the control key context:c1:interrupt
    await m.setContextVar('c1', 'interrupt', 'just-a-value')
    expect(await m.getContextVar('c1', 'interrupt')).toBe('just-a-value')
  })
})
