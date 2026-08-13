import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { ToolDefinition } from '../types/tool'

function makeConfig(overrides: Partial<AgentConfig> = {}): AgentConfig {
  return {
    agentId: 'test-agent',
    version: '1.0.0',
    systemPrompt: 'test',
    fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 2 }] },
    model: { provider: 'test', model: 'test-model', adapter: 'test' },
    ...overrides,
  }
}

class LoopGateway implements IModelGateway {
  requests = 0
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.requests++
    return {
      content: [{ type: 'text', text: 'still going' }],
      toolCalls: [{ id: `t${this.requests}`, name: 'note', input: { n: this.requests } }],
      finishReason: 'tool_use',
    }
  }
  async *stream(): AsyncIterable<never> { yield* [] }
}

class TextGateway implements IModelGateway {
  async complete(): Promise<ModelResponse> {
    return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'stop' }
  }
  async *stream(): AsyncIterable<never> { yield* [] }
}

class FailGateway implements IModelGateway {
  async complete(): Promise<ModelResponse> {
    throw Object.assign(new Error('connect failed'), { code: 'ECONNREFUSED' })
  }
  async *stream(): AsyncIterable<never> { yield* [] }
}

const noteTool: ToolDefinition = {
  name: 'note',
  description: 'note',
  inputSchema: { type: 'object', properties: { n: { type: 'number' } } },
  handler: async (input, ctx) => {
    ctx.recordArtifact?.({
      name: 'note',
      type: 'file',
      path: `note-${(input as { n: number }).n}.txt`,
    })
    return { ok: true }
  },
}

describe('#244 stopReason envelope via invoke', () => {
  it('max_iterations stops as budget_exhausted with checkpoint, not error', async () => {
    const gw = new LoopGateway()
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway: gw,
      tools: [noteTool],
    })
    milkie.registerAgent(makeConfig())

    const result = await milkie.invoke({
      agentId: 'test-agent',
      goal: 'g',
      input: 'i',
    })

    expect(result.status).not.toBe('error')
    expect(result.stopReason).toBe('budget_exhausted')
    expect(result.stopCode).toMatch(/MAX_ITERATIONS_EXCEEDED/)
    expect(result.partial).toBe(true)
    expect(result.checkpointId).toBeTruthy()
    expect(result.artifacts.length).toBeGreaterThanOrEqual(1)
    expect(gw.requests).toBe(2)

    const terminal = (await eventStore.readByRunId(result.agentRunId))
      .find(e => e.type === 'agent.run.completed')
    expect(terminal?.payload).toMatchObject({
      status: 'completed',
      stopReason: 'budget_exhausted',
    })

    const after = gw.requests
    expect(after).toBe(2)
  })

  it('infra failure is runtime_error, distinguishable from budget_exhausted', async () => {
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: new FailGateway(),
    })
    milkie.registerAgent(makeConfig({ fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5 }] } }))

    const result = await milkie.invoke({ agentId: 'test-agent', goal: 'g', input: 'i' })
    expect(result.status).toBe('error')
    expect(result.stopReason).toBe('runtime_error')
    expect(result.stopReason).not.toBe('budget_exhausted')
  })

  it('natural model stop is model_stop and not partial without contract', async () => {
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: new TextGateway(),
    })
    milkie.registerAgent(makeConfig())
    const result = await milkie.invoke({ agentId: 'test-agent', goal: 'g', input: 'i' })
    expect(result.status).toBe('completed')
    expect(result.stopReason).toBe('model_stop')
    expect(result.partial).toBe(false)
    expect(result.artifacts).toEqual([])
  })

  it('finalize hook runs after budget stop and hook failure keeps budget_exhausted', async () => {
    let hooked = 0
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: new LoopGateway(),
      tools: [noteTool],
    })
    milkie.registerAgent(makeConfig({
      onBudgetFinalize: async () => {
        hooked++
        throw new Error('hook boom')
      },
    }))
    const result = await milkie.invoke({ agentId: 'test-agent', goal: 'g', input: 'i' })
    expect(hooked).toBe(1)
    expect(result.stopReason).toBe('budget_exhausted')
    expect(result.status).toBe('completed')
    expect(result.stopCode).toMatch(/FINALIZE_FAILED/)
  })
})
