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
})
