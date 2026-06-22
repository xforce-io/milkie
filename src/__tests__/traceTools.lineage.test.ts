import { MemoryEventStore } from '../trace/MemoryEventStore'
import { makeTraceTools } from '../tools/trace'
import type { Event } from '../trace/types'

function startedEvent(runId: string, previousRunId?: string): Event {
  return { id: `${runId}-s`, runId, type: 'agent.run.started', actor: 'test', timestamp: 0,
    payload: { agentId: 'a', goal: 'g', input: 'i', contextId: 'c',
               ...(previousRunId ? { previousRunId } : {}) } } as Event
}
function objCreated(runId: string, objectId: string, type: string, meta?: Record<string, unknown>): Event {
  return { id: `${objectId}-c`, runId, type: 'object.created', actor: 'test', timestamp: 1,
    payload: { objectId, type, producerEventId: 'p', ...(meta ? { meta } : {}) } } as Event
}
function citesRel(runId: string, from: string, to: string): Event {
  return { id: `${from}-${to}`, runId, type: 'relation.created', actor: 'test', timestamp: 2,
    payload: { relationId: `${from}-${to}`, type: 'cites', fromObjectId: from, toObjectId: to, causedByEventId: 'x' } } as Event
}

describe('#189 get_lineage', () => {
  it('finds a claim in an earlier window run by query, tagged with its runId', async () => {
    const store = new MemoryEventStore()
    // run1 produced the cited number; run2 is the immediately previous run
    for (const e of [startedEvent('run1'),
      objCreated('run1', 'src1', 'passage', { source: 'reuters' }),
      objCreated('run1', 'claim1', 'claim', { text: '抓取 228 条' }),
      citesRel('run1', 'claim1', 'src1')]) await store.append(e)
    await store.append(startedEvent('run2', 'run1'))

    const tool = makeTraceTools(store).find(t => t.name === 'get_lineage')!
    const res = await tool.handler({ query: '228', lookback: 3 }, { previousRunId: 'run2' } as never) as
      { matches: { runId: string; claim: string; sources: { objectId: string }[] }[] }
    expect(res.matches).toHaveLength(1)
    expect(res.matches[0]!.runId).toBe('run1')
    expect(res.matches[0]!.sources[0]!.objectId).toBe('src1')
  })
})
