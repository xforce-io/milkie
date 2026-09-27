import type { AddressInfo } from 'net'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { BroadcastingEventStore } from '../trace/BroadcastingEventStore'
import { createServeServer } from '../cli/serve'
import type { AgentConfig } from '../types/agent'
import type { AgentInvokeRequest, AgentResult, StopReason } from '../types/common'
import type { ChildAgentRecord } from '../types/store'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'
import { IOControlError } from '../types/model'
import type { AgentRunCompletedPayload, ToolRespondedPayload } from '../trace/types'

const reasons: StopReason[] = ['budget_exhausted', 'cancelled', 'runtime_error', 'model_stop', 'deadline', 'interrupted']
const keys = ['stopReason', 'stopCode', 'partial', 'artifacts', 'checkpointId', 'error'] as const
function expectFields(actual: unknown, result: AgentResult) {
  for (const key of keys) expect((actual as Record<string, unknown>)[key]).toEqual(result[key])
}
class CapturingMilkie extends Milkie {
  lastResult?: AgentResult
  override async invoke(request: AgentInvokeRequest) { return this.lastResult = await super.invoke(request) }
  override async resume(...args: Parameters<Milkie['resume']>) { return this.lastResult = await super.resume(...args) }
}
function setup(reason: StopReason, withChild = false) {
  let mode = reason
  const counts = new Map<string, number>()
  const stateStore = new MemoryStore()
  const eventStore = new BroadcastingEventStore(new MemoryEventStore())
  const gateway: IModelGateway = {
    async complete(req: ModelRequest): Promise<ModelResponse> {
      const n = counts.get(req.model) ?? 0; counts.set(req.model, n + 1)
      if (req.model === 'parent') {
        return n === 0
          ? { content: [], toolCalls: [{ id: 'child-call', name: 'worker', input: { goal: 'g', input: 'i' } }], finishReason: 'tool_use' }
          : { content: [{ type: 'text', text: 'parent done' }], toolCalls: [], finishReason: 'stop' }
      }
      if (n > 0 && mode === 'runtime_error') throw new Error('fixture model failed after artifact')
      if (n > 0 && mode === 'model_stop') return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'stop' }
      return { content: [], toolCalls: [{ id: `note-${n}`, name: 'note', input: {} }], finishReason: 'tool_use' }
    },
    async *stream(req: ModelRequest): AsyncIterable<ModelEvent> {
      const r = await this.complete(req)
      for (const c of r.toolCalls ?? []) {
        yield { type: 'tool_call_start', data: { toolCallId: c.id, name: c.name } }
        yield { type: 'tool_call_done', data: { toolCallId: c.id, input: c.input } }
      }
      for (const c of r.content) if (c.type === 'text') yield { type: 'message_delta', data: { text: c.text } }
    },
  }
  const milkie = new CapturingMilkie({ stateStore, eventStore, gateway, tools: [{
    name: 'note', description: 'artifact fixture', inputSchema: { type: 'object' },
    handler: async (_, ctx) => {
      ctx.recordArtifact?.({ name: 'note', type: 'file', path: 'fixture.txt' })
      if (mode === 'cancelled') throw new IOControlError('IO_CANCELLED', 'tool')
      if (mode === 'deadline') throw new IOControlError('IO_DEADLINE_EXCEEDED', 'tool')
      if (mode === 'interrupted') {
        const children = await stateStore.get('context:case:children') as ChildAgentRecord[] | undefined
        await milkie.interrupt(withChild ? children![0]!.contextId! : 'case')
      }
      return 'noted'
    },
  }] })
  const config = (id: string, max: number): AgentConfig => ({ agentId: id, version: '1', systemPrompt: id,
    model: { provider: 'stub', model: id, adapter: 'stub' }, fsm: { states: [{ name: 'react', type: 'llm', max_iterations: max }] } })
  function register() {
    milkie.registerAgent(config('worker', mode === 'budget_exhausted' ? 1 : 4))
    if (withChild) milkie.registerAgent({ ...config('parent', 3), subAgents: { worker: '1' } })
  }
  register()
  return { milkie, eventStore, stateStore, reset(value: StopReason) { mode = value; counts.clear(); register() } }
}

test.each(reasons)('child %s preserves its complete result at every parent/child exit', async reason => {
  const { milkie, eventStore, stateStore } = setup(reason, true)
  const parent = await milkie.invoke({ agentId: 'parent', goal: 'g', input: 'i', contextId: 'case' })
  expect(parent.stopReason).toBe('model_stop')
  const parentEvents = await eventStore.readByRunId(parent.agentRunId)
  const response = parentEvents.find(e => e.type === 'tool.responded')!.payload as ToolRespondedPayload
  const result = response.output as AgentResult
  expect(result.stopReason).toBe(reason)
  expect(result.artifacts).toEqual([expect.objectContaining({ name: 'note' })])
  expect(result.checkpointId).toBeTruthy()
  const children = await stateStore.get('context:case:children') as ChildAgentRecord[]
  expectFields(children[0], result)
  expectFields(parentEvents.find(e => e.type === 'agent.returned')!.payload, result)
  const childEvents = await eventStore.readByRunId(result.agentRunId)
  const terminal = childEvents.find(e => e.type === 'agent.run.completed')!.payload as AgentRunCompletedPayload
  expectFields(terminal, result)
  expect(terminal.status).toBe(result.status)
  expect(terminal.lastTextOutput).toBe(result.output)
  expect(response.status).toBe('ok') // transport success, independent of child stop reason
  if (reason === 'runtime_error') expect(result.error).toBeDefined()
  // Parent replay consumes the recorded structured result without re-running the child.
  const replay = await milkie.replay(parent.agentRunId)
  expect(replay.output).toBe(parent.output)
})

for (const route of ['/chat', '/resume']) {
  test.each(reasons)(`${route} %s SSE matches SDK result and persisted terminal`, async reason => {
    const fixture = setup(reason)
    if (route === '/resume') {
      fixture.reset('model_stop')
      await fixture.milkie.invoke({ agentId: 'worker', goal: 'g', input: 'seed', contextId: 'case' })
      fixture.reset(reason)
    }
    const server = createServeServer({ milkie: fixture.milkie, broadcaster: fixture.eventStore, agentId: 'worker' })
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
    try {
      const response = await fetch(`http://127.0.0.1:${(server.address() as AddressInfo).port}${route}`, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ contextId: 'case', input: 'i' }),
        signal: AbortSignal.timeout(10000),
      })
      expect(response.status).toBe(200)
      const frames = (await response.text()).split('\n\n').filter(x => x.startsWith('event: agent.run.completed\n'))
      expect(frames).toHaveLength(1)
      const terminal = JSON.parse(frames[0]!.split('\ndata: ')[1]!)
      const result = fixture.milkie.lastResult!
      expect(result.stopReason).toBe(reason)
      expectFields(terminal, result)
      expect(terminal).toMatchObject({ runId: result.agentRunId, contextId: result.contextId, output: result.output, status: result.status })
      const events = await fixture.eventStore.readByRunId(result.agentRunId)
      expectFields(events.find(e => e.type === 'agent.run.completed')!.payload, result)
      expect(result.artifacts.length).toBeGreaterThan(0)
    } finally {
      server.closeAllConnections()
      await new Promise<void>(resolve => server.close(() => resolve()))
    }
  })
}
