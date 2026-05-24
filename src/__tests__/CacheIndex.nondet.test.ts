import type {
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
