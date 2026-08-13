import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelResponse } from '../types/model'
import type { ToolDefinition } from '../types/tool'

class TextGateway implements IModelGateway {
  constructor(private readonly writePath?: string) {}
  async complete(): Promise<ModelResponse> {
    if (this.writePath) {
      return {
        content: [],
        toolCalls: [{ id: 'w1', name: 'write_note', input: { path: this.writePath } }],
        finishReason: 'tool_use',
      }
    }
    return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'stop' }
  }
  async *stream(): AsyncIterable<never> { yield* [] }
}

class ThenStopGateway implements IModelGateway {
  n = 0
  constructor(private readonly writePath: string) {}
  async complete(): Promise<ModelResponse> {
    this.n++
    if (this.n === 1) {
      return {
        content: [],
        toolCalls: [{ id: 'w1', name: 'write_note', input: { path: this.writePath } }],
        finishReason: 'tool_use',
      }
    }
    return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'stop' }
  }
  async *stream(): AsyncIterable<never> { yield* [] }
}

const writeNote: ToolDefinition = {
  name: 'write_note',
  description: 'register a file artifact',
  inputSchema: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
  handler: async (input, ctx) => {
    const path = (input as { path: string }).path
    ctx.recordArtifact?.({ name: path, type: 'file', path })
    return { wrote: path }
  },
}

function agent(overrides: Partial<AgentConfig> = {}): AgentConfig {
  return {
    agentId: 'writer',
    version: '1',
    systemPrompt: 's',
    fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5 }] },
    model: { provider: 't', model: 't', adapter: 't' },
    ...overrides,
  }
}

describe('#247 deliverable contract via invoke', () => {
  it('S1: omitted invoke uses agent default', async () => {
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway: new ThenStopGateway('report.md'),
      tools: [writeNote],
    })
    milkie.registerAgent(agent({
      deliverables: [{ name: 'report', type: 'file', path: 'report.md', required: true }],
    }))
    const result = await milkie.invoke({ agentId: 'writer', goal: 'g', input: 'i' })
    expect(result.artifacts).toEqual([
      expect.objectContaining({ name: 'report', path: 'report.md', state: 'produced' }),
    ])
    expect(result.partial).toBe(false)
    const outcomes = (await eventStore.readByRunId(result.agentRunId))
      .filter(e => e.type === 'task.outcome.recorded')
    expect(outcomes).toHaveLength(0)
  })

  it('S2: invoke list replaces agent default wholesale', async () => {
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: new ThenStopGateway('other.md'),
      tools: [writeNote],
    })
    milkie.registerAgent(agent({
      deliverables: [{ name: 'report', type: 'file', path: 'report.md' }],
    }))
    const result = await milkie.invoke({
      agentId: 'writer',
      goal: 'g',
      input: 'i',
      deliverables: [{ name: 'other', type: 'file', path: 'other.md' }],
    })
    expect(result.artifacts.map(a => a.name)).toEqual(['other'])
    expect(result.artifacts[0]).toMatchObject({ state: 'produced', path: 'other.md' })
  })

  it('S3: missing required sets partial and does not write outcome', async () => {
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway: new ThenStopGateway('draft.txt'),
      tools: [writeNote],
    })
    milkie.registerAgent(agent({
      deliverables: [{ name: 'report', type: 'file', path: 'report.md', required: true }],
    }))
    const result = await milkie.invoke({ agentId: 'writer', goal: 'g', input: 'i' })
    expect(result.partial).toBe(true)
    expect(result.artifacts).toEqual([
      expect.objectContaining({ name: 'report', state: 'missing' }),
    ])
    expect((await eventStore.readByRunId(result.agentRunId))
      .filter(e => e.type === 'task.outcome.recorded')).toHaveLength(0)
  })

  it('S4: missing optional does not mark partial', async () => {
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: new ThenStopGateway('report.md'),
      tools: [writeNote],
    })
    milkie.registerAgent(agent({
      deliverables: [
        { name: 'report', type: 'file', path: 'report.md', required: true },
        { name: 'sources', type: 'file', path: 'sources.json', required: false },
      ],
    }))
    const result = await milkie.invoke({ agentId: 'writer', goal: 'g', input: 'i' })
    expect(result.partial).toBe(false)
    expect(result.artifacts.find(a => a.name === 'sources')?.state).toBe('missing')
  })

  it('S5: no contract does not scan or mark partial for stray files', async () => {
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: new TextGateway(),
    })
    milkie.registerAgent(agent())
    const result = await milkie.invoke({ agentId: 'writer', goal: 'g', input: 'i' })
    expect(result.partial).toBe(false)
    expect(result.artifacts).toEqual([])
  })
})
