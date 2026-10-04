import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest } from '../types/model'
import type { AgentRunStartedPayload, AgentRunCompletedPayload, LlmRequestedPayload } from '../trace/types'

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

function assistantIsSendable(message: { role: string; content: Array<{ type: string; text?: string }> }): boolean {
  if (message.role !== 'assistant') return true
  return message.content.some(part =>
    (part.type === 'text' && (part.text?.length ?? 0) > 0) || part.type === 'tool_use')
}

test('resume after a provider failure does not send an empty assistant message', async () => {
  const { milkie, store, setMode } = setup()
  setMode('error')
  const failed = await milkie.invoke({ agentId: 'worker', goal: 'g', input: 'initial', contextId: 'provider-fail' })
  expect(failed.status).toBe('error')
  setMode('text')
  const resumed = await milkie.resume(failed.checkpointId!, 'worker', 'g', 'continue')
  expect(resumed.stopReason).toBe('model_stop')
  const requested = (await store.readByRunId(resumed.agentRunId)).filter(event => event.type === 'llm.requested')
  expect(requested.length).toBeGreaterThan(0)
  for (const event of requested) {
    const messages = (event.payload as LlmRequestedPayload).request.messages
    expect(messages.every(assistantIsSendable)).toBe(true)
    expect(messages.some(message => message.role === 'user')).toBe(true)
  }
})

test('resume after a budget rejection does not send an empty assistant message', async () => {
  const budgetConfig: AgentConfig = {
    ...config,
    contextBudget: { regionCaps: { currentTurn: 1 } },
  }
  const { milkie, store, setMode } = setup()
  milkie.registerAgent(budgetConfig)
  setMode('text')
  const failed = await milkie.invoke({ agentId: 'worker', goal: 'g', input: 'initial', contextId: 'budget-fail' })
  expect(failed.stopCode).toBe('CONTEXT_BUDGET_REQUIRED_REGION_EXCEEDED')
  milkie.registerAgent({ ...config })
  const resumed = await milkie.resume(failed.checkpointId!, 'worker', 'g', 'continue')
  expect(resumed.stopReason).toBe('model_stop')
  const requested = (await store.readByRunId(resumed.agentRunId)).filter(event => event.type === 'llm.requested')
  expect(requested.length).toBeGreaterThan(0)
  for (const event of requested) {
    expect((event.payload as LlmRequestedPayload).request.messages.every(assistantIsSendable)).toBe(true)
  }
})

test('resume keeps a real assistant reply and does not repeat its tool call', async () => {
  let calls = 0
  let phase: 'tool' | 'text' | 'error' | 'again' = 'tool'
  const seen: ModelRequest[] = []
  const gateway: IModelGateway = {
    async complete(request) {
      seen.push(request)
      if (phase === 'tool') {
        phase = 'text'
        return { content: [], toolCalls: [{ id: 'note-1', name: 'note', input: {} }], finishReason: 'tool_use' }
      }
      if (phase === 'text') {
        phase = 'error'
        return { content: [{ type: 'text' as const, text: 'done' }], toolCalls: [], finishReason: 'stop' }
      }
      if (phase === 'error') throw new Error('fixture failure')
      return { content: [{ type: 'text' as const, text: 'continued' }], toolCalls: [], finishReason: 'stop' }
    },
    async *stream() { yield* [] },
  }
  const store = new MemoryEventStore()
  const milkie = new Milkie({
    stateStore: new MemoryStore(),
    eventStore: store,
    gateway,
    tools: [{
      name: 'note', description: 'note', inputSchema: { type: 'object' },
      handler: async () => { calls += 1; return 'noted' },
    }],
  })
  milkie.registerAgent({ ...config, fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 2 }] } })
  const first = await milkie.invoke({ agentId: 'worker', goal: 'g', input: 'initial', contextId: 'keep-tools' })
  expect(first.stopReason).toBe('model_stop')
  expect(calls).toBe(1)
  const failed = await milkie.resume(first.checkpointId!, 'worker', 'g', 'again')
  expect(failed.status).toBe('error')
  expect(calls).toBe(1)
  phase = 'again'
  const resumed = await milkie.resume(failed.checkpointId!, 'worker', 'g', 'recover')
  expect(resumed.stopReason).toBe('model_stop')
  expect(calls).toBe(1)
  const recovery = seen.at(-1)!
  expect(recovery.messages.every(assistantIsSendable)).toBe(true)
  expect(recovery.messages.some(message =>
    message.role === 'assistant' && message.content.some(part => part.type === 'text' && part.text === 'done'),
  )).toBe(true)
})

test('missing checkpoint is rejected before starting a run', async () => {
  const { milkie, store } = setup()
  const append = jest.spyOn(store, 'append')
  await expect(milkie.resume('missing', 'worker', 'g', 'i')).rejects.toMatchObject({ code: 'CHECKPOINT_NOT_FOUND' })
  expect(append).not.toHaveBeenCalled()
})
