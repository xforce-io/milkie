import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { Event } from '../trace/types'

describe('MemoryEventStore', () => {
  it('appends and reads back by runId', async () => {
    const s = new MemoryEventStore()
    await s.append({ id: 'e1', runId: 'r', type: 'llm.responded', actor: 'a', timestamp: 1, payload: {} } as Event)
    expect(await s.readByRunId('r')).toHaveLength(1)
    expect(await s.readByRunId('other')).toHaveLength(0)
  })

  it('stores an immutable frozen snapshot (mutating the source after append does not change the stored event)', async () => {
    const s = new MemoryEventStore()
    const payload: { plan: { steps: Array<{ status: string }> } } = { plan: { steps: [{ status: 'pending' }] } }
    const ev = { id: 'e1', runId: 'r', type: 'tool.responded', actor: 'a', timestamp: 1, payload } as unknown as Event

    await s.append(ev)
    // In-place mutate the SAME object after appending — as e.g. update_step does
    // to a plan it shares with working memory and a prior tool's output.
    payload.plan.steps[0]!.status = 'done'

    const stored = (await s.readByRunId('r'))[0]!
    expect((stored.payload as typeof payload).plan.steps[0]!.status).toBe('pending')
  })
})
