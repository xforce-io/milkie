import fs from 'fs'
import os from 'os'
import path from 'path'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { SQLiteStore } from '../store/SQLiteStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { JsonlEventStore } from '../trace/JsonlEventStore'
import type { IStateStore, AgentCheckpoint } from '../types/store'
import type { IEventStore } from '../trace/EventStore'
import type { IModelGateway } from '../types/model'
import type { AgentConfig } from '../types/agent'

const config: AgentConfig = {
  agentId: 'counter', version: '1', systemPrompt: 'counter',
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 1 }] },
}
function build(stateStore: IStateStore, eventStore?: IEventStore) {
  const observed: unknown[] = []
  const gateway: IModelGateway = {
    async complete() { return { content: [], toolCalls: [{ id: 'inc', name: 'inc', input: {} }], finishReason: 'tool_use' } },
    async *stream() { yield* [] },
  }
  const milkie = new Milkie({ stateStore, eventStore, gateway, tools: [{
    name: 'inc', description: 'increment', inputSchema: { type: 'object' },
    handler: async (_, ctx) => {
      const count = Number(ctx.workingMemory.get('count') ?? 0)
      observed.push(count)
      ctx.workingMemory.set('count', count + 1)
      return count + 1
    },
  }] })
  milkie.registerAgent(config)
  return { milkie, observed }
}
const invoke = (milkie: Milkie) => milkie.invoke({ agentId: 'counter', goal: 'g', input: 'i', contextId: 'session' })
const resume = (milkie: Milkie, id: string) => milkie.resume(id, 'counter', 'g', 'continue')

test('returned ID selects its exact snapshot even after later snapshots in the same run and context', async () => {
  const events = new MemoryEventStore()
  const { milkie, observed } = build(new MemoryStore(), events)
  const first = await invoke(milkie)
  expect(first.stopReason).toBe('budget_exhausted')
  expect(first.checkpointId).toBeTruthy()
  const cpEvent = (await events.readByRunId(first.agentRunId)).find(e => e.type === 'agent.checkpoint')!
  const cp = JSON.parse(JSON.stringify((cpEvent.payload as { checkpoint: AgentCheckpoint }).checkpoint)) as AgentCheckpoint
  cp.checkpointId = cp.checkpointId.replace(/[^:]+$/, 'later')
  cp.context.workingMemory = { data: { count: 99 }, log: [] }
  await events.append({ ...cpEvent, id: 'later-event', payload: { checkpoint: cp } })
  await invoke(milkie) // latest context now points to a different run, with count=100
  const result = await resume(milkie, first.checkpointId!)
  expect(result.stopReason).toBe('budget_exhausted')
  expect(observed).toEqual([0, 99, 1])
  await expect(resume(milkie, first.checkpointId! + '-missing')).rejects.toMatchObject({ code: 'CHECKPOINT_NOT_FOUND' })
})

test('durable snapshot ID survives SQLite/JSONL instance reconstruction and needs no state index', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-checkpoint-'))
  let state: SQLiteStore | undefined
  try {
    state = new SQLiteStore({ path: path.join(dir, 'state.sqlite') }); await state.init()
    const first = await invoke(build(state, new JsonlEventStore(path.join(dir, 'runs'))).milkie)
    state.close()
    state = new SQLiteStore({ path: path.join(dir, 'state.sqlite') }); await state.init()
    const rebuilt = build(state, new JsonlEventStore(path.join(dir, 'runs')))
    await resume(rebuilt.milkie, first.checkpointId!)
    expect(rebuilt.observed).toEqual([1])
    const withoutIndex = build(new MemoryStore(), new JsonlEventStore(path.join(dir, 'runs')))
    await resume(withoutIndex.milkie, first.checkpointId!)
    expect(withoutIndex.observed).toEqual([1])
  } finally { state?.close(); fs.rmSync(dir, { recursive: true, force: true }) }
})

test('portable session retains resolvability of exported checkpoint IDs', async () => {
  const source = build(new MemoryStore(), new MemoryEventStore())
  const result = await invoke(source.milkie)
  const target = build(new MemoryStore(), new MemoryEventStore())
  await target.milkie.importSession(await source.milkie.exportSession(result.contextId))
  await resume(target.milkie, result.checkpointId!)
  expect(target.observed).toEqual([1])
})

test('no eventStore means no advertised checkpoint; failed lookup has a stable code', async () => {
  const { milkie } = build(new MemoryStore())
  const result = await invoke(milkie)
  expect(result.checkpointId).toBeUndefined()
  await expect(resume(milkie, 'missing')).rejects.toMatchObject({ code: 'CHECKPOINT_NOT_FOUND' })
})

test('malformed or path-escaping precise IDs do not read event files', async () => {
  const events = new MemoryEventStore()
  const read = jest.spyOn(events, 'readByRunId')
  const { milkie } = build(new MemoryStore(), events)
  for (const id of ['checkpoint:v1:bad%', 'checkpoint:v1:%ZZ:x', 'checkpoint:v1:..%2Foutside:x', 'checkpoint:v1:..%5Coutside:x']) {
    await expect(resume(milkie, id)).rejects.toMatchObject({ code: 'CHECKPOINT_NOT_FOUND' })
  }
  expect(read).not.toHaveBeenCalled()
})
