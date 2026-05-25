import { BroadcastingEventStore } from '../trace/broadcast-event-store'
import { MemoryEventStore } from '../../../src/trace/MemoryEventStore'
import type { Event } from '../../../src/trace/types'

const startedEvent = (runId: string, contextId: string): Event => ({
  id: `${runId}-start`, runId, type: 'agent.run.started', actor: 'runtime', timestamp: 1,
  payload: { agentId: 'a', goal: 'g', input: 'i', contextId },
})
const llmEvent = (runId: string, id: string): Event => ({
  id, runId, type: 'llm.requested', actor: 'runtime', timestamp: 2,
  payload: { request: {}, requestHash: 'h' },
})

describe('BroadcastingEventStore', () => {
  it('forwards append to inner store', async () => {
    const inner = new MemoryEventStore()
    const store = new BroadcastingEventStore(inner)
    await store.append(startedEvent('r1', 'ctx1'))
    expect(await inner.readByRunId('r1')).toHaveLength(1)
  })

  it('broadcasts subsequent events to subscribers of the matching contextId', async () => {
    const store = new BroadcastingEventStore(new MemoryEventStore())
    const received: Event[] = []
    const unsubscribe = store.subscribe('ctx1', e => { received.push(e) })

    await store.append(startedEvent('r1', 'ctx1'))
    await store.append(llmEvent('r1', 'evt-2'))

    expect(received).toHaveLength(2)
    expect(received[0]!.id).toBe('r1-start')
    expect(received[1]!.id).toBe('evt-2')

    unsubscribe()
  })

  it('does NOT broadcast events of a different contextId to subscriber', async () => {
    const store = new BroadcastingEventStore(new MemoryEventStore())
    const receivedA: Event[] = []
    const receivedB: Event[] = []
    store.subscribe('ctxA', e => { receivedA.push(e) })
    store.subscribe('ctxB', e => { receivedB.push(e) })

    await store.append(startedEvent('runA', 'ctxA'))
    await store.append(llmEvent('runA', 'a-evt-2'))
    await store.append(startedEvent('runB', 'ctxB'))
    await store.append(llmEvent('runB', 'b-evt-2'))

    expect(receivedA.map(e => e.id).sort()).toEqual(['a-evt-2', 'runA-start'])
    expect(receivedB.map(e => e.id).sort()).toEqual(['b-evt-2', 'runB-start'])
  })

  it('unsubscribe stops further deliveries', async () => {
    const store = new BroadcastingEventStore(new MemoryEventStore())
    const received: Event[] = []
    const unsub = store.subscribe('ctx1', e => { received.push(e) })

    await store.append(startedEvent('r1', 'ctx1'))
    unsub()
    await store.append(llmEvent('r1', 'evt-2'))

    expect(received).toHaveLength(1)
  })

  it('multiple subscribers on same contextId all receive', async () => {
    const store = new BroadcastingEventStore(new MemoryEventStore())
    const a: Event[] = []
    const b: Event[] = []
    store.subscribe('ctx1', e => { a.push(e) })
    store.subscribe('ctx1', e => { b.push(e) })

    await store.append(startedEvent('r1', 'ctx1'))

    expect(a).toHaveLength(1)
    expect(b).toHaveLength(1)
  })

  it('forwards readByRunId / readRange to inner', async () => {
    const inner = new MemoryEventStore()
    const store = new BroadcastingEventStore(inner)
    await store.append(startedEvent('r1', 'ctx1'))
    await store.append(llmEvent('r1', 'evt-2'))

    expect(await store.readByRunId('r1')).toHaveLength(2)
    expect(await store.readRange('r1', 1)).toHaveLength(1)
  })
})
