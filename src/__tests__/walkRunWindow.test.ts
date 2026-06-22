import { MemoryEventStore } from '../trace/MemoryEventStore'
import { walkRunWindow } from '../trace/diagnostics/walkRunWindow'
import type { Event } from '../trace/types'

function startedEvent(runId: string, previousRunId?: string): Event {
  return {
    id: `${runId}-start`, runId, type: 'agent.run.started', actor: 'test', timestamp: 0,
    payload: { agentId: 'a', goal: 'g', input: 'i', contextId: 'c',
               ...(previousRunId ? { previousRunId } : {}) },
  } as Event
}

describe('#189 walkRunWindow', () => {
  it('walks the previousRunId chain newest→oldest, bounded by lookback', async () => {
    const store = new MemoryEventStore()
    await store.append(startedEvent('run1'))
    await store.append(startedEvent('run2', 'run1'))
    await store.append(startedEvent('run3', 'run2'))

    const w3 = await walkRunWindow(store, 'run3', 3)
    expect(w3.map(r => r.runId)).toEqual(['run3', 'run2', 'run1'])

    const w2 = await walkRunWindow(store, 'run3', 2)
    expect(w2.map(r => r.runId)).toEqual(['run3', 'run2'])
  })

  it('returns [] for undefined start and stops at a missing run', async () => {
    const store = new MemoryEventStore()
    await store.append(startedEvent('run2', 'run-missing'))
    expect(await walkRunWindow(store, undefined, 3)).toEqual([])
    const w = await walkRunWindow(store, 'run2', 3)
    expect(w.map(r => r.runId)).toEqual(['run2']) // run-missing 无事件,优雅停止
  })
})
