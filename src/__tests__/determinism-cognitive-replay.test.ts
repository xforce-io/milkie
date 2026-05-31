// SPIKE(#73): a run that uses the cognitive tools (think / create_plan) must
// replay deterministically once tool WM side-effects are event-sourced.
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { cognitiveTools } from '../tools/cognitive'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'

function toolResp(id: string, name: string, input: unknown): ModelResponse {
  return { content: [{ type: 'tool_use', id, name, input }], toolCalls: [{ id, name, input }], finishReason: 'tool_use' }
}

class ScriptedGateway implements IModelGateway {
  private n = 0
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.n++
    if (this.n === 1) return toolResp('p1', 'create_plan', { steps: ['a', 'b'] })
    if (this.n === 2) return toolResp('t1', 'think', { thoughts: 'reason' })
    return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'end_turn' }
  }
  async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] }
}

const agent: AgentConfig = {
  agentId: 'planner', version: '1.0.0',
  systemPrompt: 'plan then think then answer',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 10 }] },
  model: { provider: 'test', model: 'test', adapter: 'test' },
}

describe('determinism: cognitive-tool runs replay without divergence', () => {
  it('a completed think/create_plan run replays deterministically', async () => {
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway:    new ScriptedGateway(),
      tools:      cognitiveTools,
    })
    milkie.registerAgent(agent)

    const run = await milkie.invoke({ agentId: 'planner', goal: 'g', input: 'go', contextId: 'det-cog' })
    expect(run.status).toBe('completed')

    await expect(milkie.replay(run.agentRunId)).resolves.toMatchObject({ output: run.output })
  })
})
