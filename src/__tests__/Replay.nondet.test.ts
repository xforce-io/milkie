import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { JsonlEventStore } from '../trace/JsonlEventStore'
import { ReplayDivergenceError } from '../trace/ReplayDivergenceError'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { Event } from '../trace/types'
import fs from 'fs'
import os from 'os'
import path from 'path'

class SequentialGateway implements IModelGateway {
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.responses.shift()
    if (!r) throw new Error('SequentialGateway exhausted')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}

const text = (s: string): ModelResponse => ({
  content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn',
})

const echoAgentMd = `---
agentId: echo
fsm:
  states:
    - name: react
      type: llm
      instructions: say hi
      tools: []
model:
  provider: stub
  model: stub
  adapter: openai-compatible
---
sys`

describe('Milkie.replay — Phase 4 tail check (P-wide)', () => {
  let tmpDir: string
  let agentFile: string
  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-p4-replay-'))
    agentFile = path.join(tmpDir, 'echo.md')
    fs.writeFileSync(agentFile, echoAgentMd)
  })
  afterEach(() => { fs.rmSync(tmpDir, { recursive: true, force: true }) })

  it('replay succeeds with no unconsumed events', async () => {
    const eventStore = new JsonlEventStore(path.join(tmpDir, 'runs'))
    const record = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([text('hello')]), eventStore })
    record.loadAgentFile(agentFile)
    const original = await record.invoke({ agentId: 'echo', goal: 'g', input: 'i' })
    expect(original.status).toBe('completed')

    const replayer = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([]), eventStore })
    replayer.loadAgentFile(agentFile)
    const replayed = await replayer.replay(original.agentRunId)
    expect(replayed.status).toBe('completed')
  })

  it('replay throws ReplayDivergenceError when recorded clock events go unconsumed', async () => {
    const eventStore = new JsonlEventStore(path.join(tmpDir, 'runs'))
    const record = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([text('hello')]), eventStore })
    record.loadAgentFile(agentFile)
    const original = await record.invoke({ agentId: 'echo', goal: 'g', input: 'i' })

    const phantom: Event = {
      id: 'phantom-clock', runId: original.agentRunId, type: 'clock.read',
      actor: 'runtime', timestamp: 999, payload: { value: 999 },
    }
    fs.appendFileSync(
      path.join(tmpDir, 'runs', `${original.agentRunId}.jsonl`),
      JSON.stringify(phantom) + '\n',
    )

    const replayer = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([]), eventStore })
    replayer.loadAgentFile(agentFile)
    await expect(replayer.replay(original.agentRunId)).rejects.toThrow(ReplayDivergenceError)
    try { await replayer.replay(original.agentRunId) }
    catch (err) {
      const e = err as ReplayDivergenceError
      expect(e.kind).toBe('clock')
      expect(e.message).toContain('unconsumed')
    }
  })

  it('replay throws ReplayDivergenceError when recorded uuid events go unconsumed', async () => {
    const eventStore = new JsonlEventStore(path.join(tmpDir, 'runs'))
    const record = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([text('hello')]), eventStore })
    record.loadAgentFile(agentFile)
    const original = await record.invoke({ agentId: 'echo', goal: 'g', input: 'i' })

    const phantom: Event = {
      id: 'phantom-uuid', runId: original.agentRunId, type: 'uuid.generated',
      actor: 'runtime', timestamp: 999, payload: { value: 'never-consumed' },
    }
    fs.appendFileSync(
      path.join(tmpDir, 'runs', `${original.agentRunId}.jsonl`),
      JSON.stringify(phantom) + '\n',
    )

    const replayer = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([]), eventStore })
    replayer.loadAgentFile(agentFile)
    await expect(replayer.replay(original.agentRunId)).rejects.toThrow(ReplayDivergenceError)
  })

  it('replay throws ReplayDivergenceError when recorded llm events go unconsumed', async () => {
    const eventStore = new JsonlEventStore(path.join(tmpDir, 'runs'))
    const record = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([text('hello')]), eventStore })
    record.loadAgentFile(agentFile)
    const original = await record.invoke({ agentId: 'echo', goal: 'g', input: 'i' })

    const phantom: Event = {
      id: 'phantom-llm', runId: original.agentRunId, type: 'llm.responded',
      actor: 'runtime', timestamp: 999,
      payload: { response: text('phantom'), requestHash: 'phantom-hash-never-issued' },
    }
    fs.appendFileSync(
      path.join(tmpDir, 'runs', `${original.agentRunId}.jsonl`),
      JSON.stringify(phantom) + '\n',
    )

    const replayer = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([]), eventStore })
    replayer.loadAgentFile(agentFile)
    await expect(replayer.replay(original.agentRunId)).rejects.toThrow(ReplayDivergenceError)
    try { await replayer.replay(original.agentRunId) }
    catch (err) {
      const e = err as ReplayDivergenceError
      expect(e.kind).toBe('llm')
    }
  })

  // ── Phase 4 end-to-end smoke: record/replay loop survives a real agent run
  //
  // The echo agent currently triggers exactly 1 `clock.read` per run and zero
  // `uuid.generated` (because contextId/agentRunId are minted in `Milkie.invoke`
  // before the IOPort exists, and `batchId` only fires when tools execute —
  // echo has none). That's enough to prove Phase 4 is wired end-to-end:
  // recording captures the clock, replay consumes it from cache, the tail
  // check sees zero remaining. We do NOT prove the stronger "byte-identical
  // because nondet flows into LLM request hash" property — no nondet value
  // currently affects ContextRegions.assemble. Demonstrating that requires
  // either routing contextId/agentRunId through `ioPort.uuid()` (a small
  // runtime change worth doing as a Phase 5 follow-up) or a custom multi-step
  // tool fixture. Out of scope for this task.
  it('record-then-replay round trip with Phase 4 active: recorded nondet, replay succeeds', async () => {
    const eventStore = new JsonlEventStore(path.join(tmpDir, 'runs'))
    const record = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([text('hello')]), eventStore })
    record.loadAgentFile(agentFile)
    const original = await record.invoke({ agentId: 'echo', goal: 'g', input: 'i' })
    expect(original.status).toBe('completed')

    // Recording captured at least one nondet event (echo agent triggers 1 clock.read).
    const recordedEvents = await eventStore.readByRunId(original.agentRunId)
    const nondets = recordedEvents.filter(e => e.type === 'clock.read' || e.type === 'uuid.generated')
    expect(nondets.length).toBeGreaterThan(0)

    // Replay with an empty gateway: any live LLM call would throw 'exhausted'.
    // Replay completing proves: (a) cache served the LLM response, (b) recorded
    // nondet was consumed from the queue (not silently passed through), (c)
    // tail check found zero unconsumed across all four queues.
    const replayer = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([]), eventStore })
    replayer.loadAgentFile(agentFile)
    const replayed = await replayer.replay(original.agentRunId)
    expect(replayed.status).toBe('completed')
    expect(replayed.output).toBe(original.output)
  })

  it('over-consume: tampering strips recorded clock.read → ReplayDivergenceError kind=clock', async () => {
    const eventStore = new JsonlEventStore(path.join(tmpDir, 'runs'))
    const record = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([text('hello')]), eventStore })
    record.loadAgentFile(agentFile)
    const original = await record.invoke({ agentId: 'echo', goal: 'g', input: 'i' })

    // Sanity: recording captured at least one clock.read (otherwise the strip is a no-op).
    const recordedEvents = await eventStore.readByRunId(original.agentRunId)
    expect(recordedEvents.filter(e => e.type === 'clock.read').length).toBeGreaterThan(0)

    // Tamper: strip all clock.read events. Replay's first port.now() call will
    // hit an empty queue → over-consume → ReplayDivergenceError kind='clock'.
    const filePath = path.join(tmpDir, 'runs', `${original.agentRunId}.jsonl`)
    const filtered = fs.readFileSync(filePath, 'utf-8')
      .split('\n')
      .filter(line => line.length > 0)
      .map(line => JSON.parse(line))
      .filter((e: { type: string }) => e.type !== 'clock.read')
      .map(e => JSON.stringify(e))
      .join('\n') + '\n'
    fs.writeFileSync(filePath, filtered)

    const replayer = new Milkie({ stateStore: new MemoryStore(),
      gateway: new SequentialGateway([]), eventStore })
    replayer.loadAgentFile(agentFile)
    await expect(replayer.replay(original.agentRunId)).rejects.toThrow(ReplayDivergenceError)
    try { await replayer.replay(original.agentRunId) }
    catch (err) {
      const e = err as ReplayDivergenceError
      expect(e.kind).toBe('clock')
      expect(e.message).toContain('exhausted')
    }
  })
})
