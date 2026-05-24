import { CacheIndex, CacheIndexEmptyError } from '../trace/CacheIndex'
import type {
  Event,
  ClockReadEvent,
  ClockReadPayload,
  UuidGeneratedEvent,
  UuidGeneratedPayload,
} from '../trace/types'

describe('Phase 4 event types', () => {
  it('ClockReadEvent is structurally correct', () => {
    const evt: ClockReadEvent = {
      id: 'x', runId: 'r', type: 'clock.read', actor: 'runtime', timestamp: 0,
      payload: { value: 12345 } satisfies ClockReadPayload,
    }
    expect(evt.type).toBe('clock.read')
    expect(evt.payload.value).toBe(12345)
  })

  it('UuidGeneratedEvent is structurally correct', () => {
    const evt: UuidGeneratedEvent = {
      id: 'x', runId: 'r', type: 'uuid.generated', actor: 'runtime', timestamp: 0,
      payload: { value: 'some-uuid-string' } satisfies UuidGeneratedPayload,
    }
    expect(evt.type).toBe('uuid.generated')
    expect(evt.payload.value).toBe('some-uuid-string')
  })
})

const clockEvent = (id: string, value: number): Event => ({
  id, runId: 'r', type: 'clock.read', actor: 'runtime', timestamp: 0,
  payload: { value },
})

const uuidEvent = (id: string, value: string): Event => ({
  id, runId: 'r', type: 'uuid.generated', actor: 'runtime', timestamp: 0,
  payload: { value },
})

describe('CacheIndex — clock/uuid queues', () => {
  it('consumeClock returns values in FIFO order across the entire log', () => {
    const cache = CacheIndex.fromEvents([
      clockEvent('c1', 100),
      clockEvent('c2', 200),
      clockEvent('c3', 300),
    ])
    expect(cache.consumeClock()).toBe(100)
    expect(cache.consumeClock()).toBe(200)
    expect(cache.consumeClock()).toBe(300)
  })

  it('consumeUuid returns values in FIFO order', () => {
    const cache = CacheIndex.fromEvents([
      uuidEvent('u1', 'first-uuid'),
      uuidEvent('u2', 'second-uuid'),
    ])
    expect(cache.consumeUuid()).toBe('first-uuid')
    expect(cache.consumeUuid()).toBe('second-uuid')
  })

  it('consumeClock throws CacheIndexEmptyError when queue empty', () => {
    const cache = CacheIndex.fromEvents([])
    expect(() => cache.consumeClock()).toThrow(CacheIndexEmptyError)
  })

  it('consumeUuid throws CacheIndexEmptyError when queue empty', () => {
    const cache = CacheIndex.fromEvents([clockEvent('c1', 1)])
    expect(() => cache.consumeUuid()).toThrow(CacheIndexEmptyError)
  })

  it('remaining() reports all four queues including clock + uuid', () => {
    const cache = CacheIndex.fromEvents([
      clockEvent('c1', 100),
      clockEvent('c2', 200),
      uuidEvent('u1', 'a'),
    ])
    const r = cache.remaining()
    expect(r).toEqual({ llm: 0, tool: 0, clock: 2, uuid: 1 })
  })

  it('remaining decreases as values are consumed', () => {
    const cache = CacheIndex.fromEvents([
      clockEvent('c1', 100),
      uuidEvent('u1', 'a'),
    ])
    cache.consumeClock()
    cache.consumeUuid()
    expect(cache.remaining()).toEqual({ llm: 0, tool: 0, clock: 0, uuid: 0 })
  })

  it('clock and uuid queues do not interfere with each other', () => {
    const cache = CacheIndex.fromEvents([
      clockEvent('c1', 100),
      uuidEvent('u1', 'a'),
      clockEvent('c2', 200),
      uuidEvent('u2', 'b'),
    ])
    expect(cache.consumeClock()).toBe(100)
    expect(cache.consumeUuid()).toBe('a')
    expect(cache.consumeClock()).toBe(200)
    expect(cache.consumeUuid()).toBe('b')
  })
})
