import { RecordingIOPort } from '../trace/RecordingIOPort'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { IIOPort } from '../runtime/IOPort'
import type { ModelRequest, ModelResponse } from '../types/model'

class StubInnerPort implements IIOPort {
  public clockCalls = 0
  public uuidCalls  = 0
  private nextClock = 1000
  private nextUuid  = 1

  async invokeLLM(_req: ModelRequest): Promise<ModelResponse> {
    return { content: [{ type: 'text', text: 'stub' }], toolCalls: [], finishReason: 'end_turn' }
  }
  async invokeTool(_n: string, _i: unknown, _e: () => Promise<unknown>): Promise<unknown> {
    return 'stub-output'
  }
  now():  number { this.clockCalls++; return this.nextClock++ }
  uuid(): string { this.uuidCalls++; return `uuid-${this.nextUuid++}` }
}

describe('RecordingIOPort — non-determinism recording', () => {
  it('now() returns inner value and queues a clock.read event', async () => {
    const inner = new StubInnerPort()
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r1')

    const val = port.now()
    expect(typeof val).toBe('number')

    // event not flushed yet — flush only happens at next async method entry
    expect((await store.readByRunId('r1')).filter(e => e.type === 'clock.read')).toHaveLength(0)

    // trigger flush via any async method
    await port.detach({ status: 'completed' })

    const clockEvents = (await store.readByRunId('r1')).filter(e => e.type === 'clock.read')
    expect(clockEvents).toHaveLength(1)
    expect((clockEvents[0]!.payload as { value: number }).value).toBe(val)
  })

  it('uuid() returns inner value and queues a uuid.generated event', async () => {
    const inner = new StubInnerPort()
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r1')

    const val = port.uuid()
    expect(typeof val).toBe('string')

    await port.detach({ status: 'completed' })

    const uuidEvents = (await store.readByRunId('r1')).filter(e => e.type === 'uuid.generated')
    expect(uuidEvents).toHaveLength(1)
    expect((uuidEvents[0]!.payload as { value: string }).value).toBe(val)
  })

  it('multiple sync now/uuid calls flush in input order at next async boundary', async () => {
    const inner = new StubInnerPort()
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r1')

    port.now()
    port.uuid()
    port.now()
    port.uuid()

    await port.attach({ agentId: 'a', goal: 'g', input: 'i', contextId: 'r1' })

    // events should appear in interleaved order BEFORE agent.run.started
    const events = await store.readByRunId('r1')
    const nondetIdx = events
      .map((e, i) => ['clock.read', 'uuid.generated'].includes(e.type) ? i : -1)
      .filter(i => i >= 0)
    const startedIdx = events.findIndex(e => e.type === 'agent.run.started')
    expect(nondetIdx).toHaveLength(4)
    for (const i of nondetIdx) expect(i).toBeLessThan(startedIdx)
    expect(events[nondetIdx[0]!]!.type).toBe('clock.read')
    expect(events[nondetIdx[1]!]!.type).toBe('uuid.generated')
    expect(events[nondetIdx[2]!]!.type).toBe('clock.read')
    expect(events[nondetIdx[3]!]!.type).toBe('uuid.generated')
  })

  it('flush happens at every async method entry, not only detach', async () => {
    const inner = new StubInnerPort()
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r1')

    await port.attach({ agentId: 'a', goal: 'g', input: 'i', contextId: 'r1' })

    port.now()
    await port.invokeLLM({ provider: 'stub', model: 'stub', messages: [] } as ModelRequest)

    // clock.read recorded BEFORE llm.requested (flushed at invokeLLM entry)
    const events = await store.readByRunId('r1')
    const clockIdx = events.findIndex(e => e.type === 'clock.read')
    const llmReqIdx = events.findIndex(e => e.type === 'llm.requested')
    expect(clockIdx).toBeGreaterThan(-1)
    expect(llmReqIdx).toBeGreaterThan(-1)
    expect(clockIdx).toBeLessThan(llmReqIdx)
  })

  it('infrastructure now/uuid (for event id/timestamp) do not recurse into nondet events', async () => {
    const inner = new StubInnerPort()
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r1')

    // attach + detach use inner.now/uuid internally for event id/timestamp,
    // but those must NOT be recorded as clock.read/uuid.generated.
    await port.attach({ agentId: 'a', goal: 'g', input: 'i', contextId: 'r1' })
    await port.detach({ status: 'completed' })

    const events  = await store.readByRunId('r1')
    const nondets = events.filter(e => e.type === 'clock.read' || e.type === 'uuid.generated')
    expect(nondets).toHaveLength(0)   // zero agent-facing calls were made
    // sanity: inner WAS called (for event id/timestamp on attach + detach)
    expect(inner.clockCalls).toBeGreaterThan(0)
    expect(inner.uuidCalls).toBeGreaterThan(0)
  })
})
