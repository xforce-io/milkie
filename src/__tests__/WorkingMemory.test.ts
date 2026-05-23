import { WorkingMemory } from '../store/WorkingMemory'

describe('WorkingMemory', () => {
  it('get / set / has / delete', () => {
    const wm = new WorkingMemory()
    expect(wm.has('key')).toBe(false)
    wm.set('key', { value: 42 })
    expect(wm.has('key')).toBe(true)
    expect(wm.get('key')).toEqual({ value: 42 })
    wm.delete('key')
    expect(wm.has('key')).toBe(false)
  })

  it('append adds to log', () => {
    const wm = new WorkingMemory()
    wm.append({ type: 'thought', content: 'hello' })
    expect(wm.getLog()).toHaveLength(1)
    expect(wm.getLog()[0]?.type).toBe('thought')
  })

  it('serialises and deserialises round-trip', () => {
    const wm = new WorkingMemory()
    wm.set('plan', { id: '1', steps: [] })
    wm.append({ type: 'thought', content: 'reasoning' })

    const json = wm.toJSON()
    const wm2  = WorkingMemory.fromJSON(json)

    expect(wm2.get('plan')).toEqual({ id: '1', steps: [] })
    expect(wm2.getLog()).toHaveLength(1)
  })
})
