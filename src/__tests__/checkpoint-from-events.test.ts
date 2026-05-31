// #73 Stage 6: the resume state is in the event log (agent.checkpoint event).
// Differential proof: the checkpoint projected from events deep-equals the
// legacy stateStore checkpoint blob — so the stateStore blob is redundant.
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { checkpointFromEvents } from '../trace/diagnostics/checkpointFromEvents'
import type { AgentCheckpoint } from '../types/store'
import type { ToolDefinition } from '../types/tool'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'

function slowWriter(): ToolDefinition {
  return {
    name: 'record_fact', description: 'record a fact',
    inputSchema: { type: 'object', properties: { key: { type: 'string' }, value: { type: 'string' } }, required: ['key', 'value'] },
    handler: async (input: unknown, ctx) => {
      const { key, value } = input as { key: string; value: string }
      await new Promise<void>(r => setTimeout(r, 120))
      ctx.workingMemory.set(key, value)
      return { recorded: key }
    },
  }
}
class FactGateway implements IModelGateway {
  private n = 0
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.n++
    const input = { key: `fact${this.n}`, value: `v${this.n}` }
    return { content: [{ type: 'tool_use', id: `c${this.n}`, name: 'record_fact', input }], toolCalls: [{ id: `c${this.n}`, name: 'record_fact', input }], finishReason: 'tool_use' }
  }
  async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] }
}
const agent: AgentConfig = {
  agentId: 'recorder', version: '1.0.0', systemPrompt: 'record facts',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 50 }] },
  model: { provider: 'test', model: 'test', adapter: 'test' },
}

describe('#73 Stage 6: checkpoint projected from the event log == stateStore checkpoint', () => {
  it('emits an agent.checkpoint event on interrupt, equal to the stateStore blob', async () => {
    const stateStore = new MemoryStore()
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({ stateStore, eventStore, gateway: new FactGateway(), tools: [slowWriter()] })
    milkie.registerAgent(agent)

    const contextId = 'cp'
    const runP = milkie.invoke({ agentId: 'recorder', goal: 'g', input: 'go', contextId })
    await new Promise<void>(r => setTimeout(r, 320))
    await milkie.interrupt(contextId)
    const run = await runP
    expect(run.status).toBe('interrupted')

    const fromStore = await stateStore.get(`context:${contextId}:checkpoint:latest`) as AgentCheckpoint
    const events = await eventStore.readByRunId(run.agentRunId)
    const fromEvents = checkpointFromEvents(events)

    expect(fromEvents).not.toBeNull()
    // The event-sourced checkpoint reproduces the full stateStore blob in its
    // DURABLE form. (A real stateStore — SQLiteStore — JSON-serialises on set,
    // so the checkpoint is already round-tripped through JSON in production;
    // the in-memory MemoryStore keeps references, hence the JSON normalisation.)
    expect(fromEvents).toEqual(JSON.parse(JSON.stringify(fromStore)))
    // And the context→runId routing pointer is in place.
    expect(await stateStore.get(`context:${contextId}:checkpoint-run:latest`)).toBe(run.agentRunId)
  })

  it('multi-turn resume restores WM from the event log even with the stateStore blob deleted', async () => {
    const stateStore = new MemoryStore()
    const eventStore = new MemoryEventStore()
    // First gateway records facts then we interrupt; a fresh gateway for the
    // resumed turn just answers immediately.
    const milkie = new Milkie({ stateStore, eventStore, gateway: new FactGateway(), tools: [slowWriter()] })
    milkie.registerAgent(agent)

    const contextId = 'mt'
    const runP = milkie.invoke({ agentId: 'recorder', goal: 'g', input: 'go', contextId })
    await new Promise<void>(r => setTimeout(r, 320))
    await milkie.interrupt(contextId)
    await runP

    // Delete the legacy stateStore checkpoint blob → only the event log can restore.
    await stateStore.delete(`context:${contextId}:checkpoint:latest`)

    // Resume turn: answer immediately so it completes.
    const m2 = new Milkie({
      stateStore, eventStore,
      gateway: { async complete() { return { content: [{ type: 'text', text: 'resumed' }], toolCalls: [], finishReason: 'end_turn' } }, async *stream() { yield* [] } } as IModelGateway,
      tools: [slowWriter()],
    })
    m2.registerAgent(agent)
    const resumed = await m2.invoke({ agentId: 'recorder', goal: 'g', input: 'continue', contextId })

    // The resumed run's first prompt must carry the prior turn's WM (restored
    // from the event log, not the deleted stateStore blob).
    const events = await eventStore.readByRunId(resumed.agentRunId)
    const firstLlm = events.find(e => e.type === 'llm.requested')!
    const prompt = JSON.stringify((firstLlm.payload as { request: unknown }).request)
    expect(prompt).toContain('fact1')  // a fact recorded in the interrupted turn
  })

  it('returns null for a run that completed without interruption', async () => {
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(), eventStore,
      gateway: { async complete() { return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'end_turn' } }, async *stream() { yield* [] } } as IModelGateway,
    })
    milkie.registerAgent(agent)
    const run = await milkie.invoke({ agentId: 'recorder', goal: 'g', input: 'go', contextId: 'done' })
    expect(run.status).toBe('completed')
    expect(checkpointFromEvents(await eventStore.readByRunId(run.agentRunId))).toBeNull()
  })
})
