// #73 Stage 4 (Gate 3): the working-memory state captured in the event log is
// equivalent to the state a checkpoint stores. This is the load-bearing claim
// behind "checkpoint is redundant for WM" — the only field a checkpoint holds
// that the event log otherwise could not reconstruct.
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { checkpointFromEvents } from '../trace/diagnostics/checkpointFromEvents'
import type { ToolDefinition } from '../types/tool'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'

// A WM-writing tool slow enough to let an interrupt land between tool calls.
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

// Keeps asking to record facts (never finishes on its own → we interrupt).
class FactGateway implements IModelGateway {
  private n = 0
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.n++
    const id = `c${this.n}`
    const input = { key: `fact${this.n}`, value: `v${this.n}` }
    return { content: [{ type: 'tool_use', id, name: 'record_fact', input }], toolCalls: [{ id, name: 'record_fact', input }], finishReason: 'tool_use' }
  }
  async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] }
}

const agent: AgentConfig = {
  agentId: 'recorder', version: '1.0.0', systemPrompt: 'record facts',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 50 }] },
  model: { provider: 'test', model: 'test', adapter: 'test' },
}

describe('#73 Stage 4 (Gate 3): event-log WM state == checkpoint WM state', () => {
  it('the WM in the interrupt checkpoint equals the WM reconstructed from the event log', async () => {
    const stateStore = new MemoryStore()
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({ stateStore, eventStore, gateway: new FactGateway(), tools: [slowWriter()] })
    milkie.registerAgent(agent)

    const contextId = 'eq'
    const runP = milkie.invoke({ agentId: 'recorder', goal: 'g', input: 'go', contextId })
    await new Promise<void>(r => setTimeout(r, 320))  // let ~2 facts get recorded
    await milkie.interrupt(contextId)
    const run = await runP
    expect(run.status).toBe('interrupted')

    // (A) WM the resume checkpoint stores — now an agent.checkpoint EVENT.
    const events = await eventStore.readByRunId(run.agentRunId)
    const cp = checkpointFromEvents(events)!
    const cpWm = cp.context.workingMemory as { data: Record<string, unknown> }

    // (B) WM reconstructed from the event log = the latest wm.mutated snapshot.
    const wmEvents = events.filter(e => e.type === 'wm.mutated')
    expect(wmEvents.length).toBeGreaterThan(0)
    const eventWm = (wmEvents[wmEvents.length - 1]!.payload as { snapshot: { data: Record<string, unknown> } }).snapshot

    // The two representations of WM are equal — so the checkpoint's WM carries
    // nothing the event log doesn't.
    expect(eventWm.data).toEqual(cpWm.data)
    // And it is non-trivial (facts were actually recorded).
    expect(Object.keys(cpWm.data).length).toBeGreaterThan(0)
  })
})
