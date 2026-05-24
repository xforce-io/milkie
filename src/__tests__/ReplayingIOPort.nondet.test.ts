import { ReplayingIOPort } from '../trace/ReplayingIOPort'
import { CacheIndex } from '../trace/CacheIndex'
import { ReplayDivergenceError } from '../trace/ReplayDivergenceError'
import type { IIOPort } from '../runtime/IOPort'
import type { ModelRequest, ModelResponse } from '../types/model'
import type { Event } from '../trace/types'

class ExplodingInnerPort implements IIOPort {
  async invokeLLM(_req: ModelRequest): Promise<ModelResponse> {
    throw new Error('inner.invokeLLM must not be called during replay')
  }
  async invokeTool(_n: string, _i: unknown, _e: () => Promise<unknown>): Promise<unknown> {
    throw new Error('inner.invokeTool must not be called during replay')
  }
  now(): number { throw new Error('inner.now must not be called during nondet replay') }
  uuid(): string { throw new Error('inner.uuid must not be called during nondet replay') }
}

const clockEvent = (value: number): Event => ({
  id: 'c', runId: 'r', type: 'clock.read', actor: 'runtime', timestamp: 0,
  payload: { value },
})
const uuidEvent = (value: string): Event => ({
  id: 'u', runId: 'r', type: 'uuid.generated', actor: 'runtime', timestamp: 0,
  payload: { value },
})

describe('ReplayingIOPort — nondet consumption', () => {
  it('now() returns cached value in FIFO order without touching inner', () => {
    const cache = CacheIndex.fromEvents([clockEvent(111), clockEvent(222)])
    const port  = new ReplayingIOPort(cache, new ExplodingInnerPort())
    expect(port.now()).toBe(111)
    expect(port.now()).toBe(222)
  })

  it('uuid() returns cached value in FIFO order without touching inner', () => {
    const cache = CacheIndex.fromEvents([uuidEvent('a'), uuidEvent('b')])
    const port  = new ReplayingIOPort(cache, new ExplodingInnerPort())
    expect(port.uuid()).toBe('a')
    expect(port.uuid()).toBe('b')
  })

  it('now() throws ReplayDivergenceError when clock queue is exhausted', () => {
    const cache = CacheIndex.fromEvents([])
    const port  = new ReplayingIOPort(cache, new ExplodingInnerPort())
    expect(() => port.now()).toThrow(ReplayDivergenceError)
    try { port.now() }
    catch (err) {
      const e = err as ReplayDivergenceError
      expect(e.kind).toBe('clock')
    }
  })

  it('uuid() throws ReplayDivergenceError when uuid queue is exhausted', () => {
    const cache = CacheIndex.fromEvents([])
    const port  = new ReplayingIOPort(cache, new ExplodingInnerPort())
    expect(() => port.uuid()).toThrow(ReplayDivergenceError)
    try { port.uuid() }
    catch (err) {
      const e = err as ReplayDivergenceError
      expect(e.kind).toBe('uuid')
    }
  })
})
