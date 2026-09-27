import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway } from '../types/model'
import type { AgentRunStartedPayload, AgentRunCompletedPayload } from '../trace/types'

const config: AgentConfig = { agentId: 'worker', version: '1', systemPrompt: 'work',
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 1 }] } }
function setup() {
  let mode = 'loop'
  const gateway: IModelGateway = {
    async complete() {
      if (mode === 'error') throw new Error('fixture failure')
      if (mode === 'text') return { content: [{ type: 'text' as const, text: 'done' }], toolCalls: [], finishReason: 'stop' }
      return { content: [], toolCalls: [{ id: 'note-1', name: 'note', input: {} }], finishReason: 'tool_use' }
    },
    async *stream() { yield* [] },
  }
  const store = new MemoryEventStore()
  const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: store, gateway, tools: [{
    name: 'note', description: 'note', inputSchema: { type: 'object' },
    handler: async (_, ctx) => { ctx.workingMemory.set('n', Number(ctx.workingMemory.get('n') ?? 0) + 1); return 'noted' },
  }] })
  milkie.registerAgent(config)
  return { milkie, store, setMode: (value: string) => { mode = value } }
}

test('three successive resumes keep their own activity, start/source, terminal and current summary', async () => {
  const { milkie, store, setMode } = setup()
  let previous = await milkie.invoke({ agentId: 'worker', goal: 'g', input: 'initial', contextId: 'ctx' })
  const ids = new Set([previous.agentRunId])
  for (const mode of ['loop', 'text', 'error']) {
    const before = await store.readByRunId(previous.agentRunId)
    setMode(mode)
    const next = await milkie.resume(previous.checkpointId!, 'worker', 'g', mode)
    expect(ids.has(next.agentRunId)).toBe(false)
    ids.add(next.agentRunId)
    expect(next.contextId).toBe('ctx')
    expect(await store.readByRunId(previous.agentRunId)).toEqual(before)
    const events = await store.readByRunId(next.agentRunId)
    const started = events.filter(e => e.type === 'agent.run.started')
    const completed = events.filter(e => e.type === 'agent.run.completed')
    expect(started).toHaveLength(1); expect(completed).toHaveLength(1)
    expect(started[0]!.payload as AgentRunStartedPayload).toMatchObject({ previousRunId: previous.agentRunId, resumedFromCheckpointId: previous.checkpointId })
    expect(completed[0]!.payload as AgentRunCompletedPayload).toMatchObject({ status: next.status, stopReason: next.stopReason, partial: next.partial, artifacts: next.artifacts, checkpointId: next.checkpointId })
    expect(events.some(e => e.type === 'llm.requested')).toBe(true)
    expect(events.at(-1)?.type).toBe('agent.run.completed')
    expect(next.stopReason).toBe(mode === 'loop' ? 'budget_exhausted' : mode === 'text' ? 'model_stop' : 'runtime_error')
    expect((await milkie.getRunSummary(next.agentRunId)).stopReason).toBe(next.stopReason)
    previous = next
  }
})

test('resumed run replay restores its exact source and survives portable session import', async () => {
  const { milkie, setMode } = setup()
  const first = await milkie.invoke({ agentId: 'worker', goal: 'g', input: 'initial', contextId: 'ctx' })
  const second = await milkie.resume(first.checkpointId!, 'worker', 'g', 'resume')
  setMode('error') // replay must not call this live gateway
  const replay = await milkie.replay(second.agentRunId)
  expect(replay.stopReason).toBe(second.stopReason)
  const session = await milkie.exportSession('ctx')
  expect(session.events.some(e => e.runId === first.agentRunId)).toBe(true)
  const fresh = setup()
  fresh.setMode('error')
  await fresh.milkie.importSession(session)
  expect((await fresh.milkie.replay(second.agentRunId)).stopReason).toBe(second.stopReason)
})

test('missing checkpoint is rejected before starting a run', async () => {
  const { milkie, store } = setup()
  const append = jest.spyOn(store, 'append')
  await expect(milkie.resume('missing', 'worker', 'g', 'i')).rejects.toMatchObject({ code: 'CHECKPOINT_NOT_FOUND' })
  expect(append).not.toHaveBeenCalled()
})
