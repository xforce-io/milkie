/**
 * Deterministic e2e tests for core framework behavior.
 *
 * These tests use scripted gateways so CI can validate Milkie semantics without
 * depending on live model tool-selection behavior.
 */

import type { AgentConfig } from '../../src/types/agent.js'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model.js'
import type { ToolDefinition } from '../../src/types/tool.js'
import { Milkie } from '../../src/runtime/Milkie.js'
import { MemoryStore } from '../../src/store/MemoryStore.js'
import { TrajectoryStore } from '../../src/trajectory/TrajectoryStore.js'

function text(text: string): ModelResponse {
  return {
    content:      [{ type: 'text', text }],
    toolCalls:    [],
    finishReason: 'end_turn',
  }
}

function tools(calls: Array<{ id: string; name: string; input: unknown }>): ModelResponse {
  return {
    content: calls.map(c => ({ type: 'tool_use' as const, id: c.id, name: c.name, input: c.input })),
    toolCalls: calls,
    finishReason: 'tool_use',
  }
}

const MODEL = {
  provider: 'test',
  model:    'test-model',
  adapter:  'test',
}

class SequentialGateway implements IModelGateway {
  private index = 0

  constructor(private readonly responses: ModelResponse[]) {}

  async complete(_request: ModelRequest): Promise<ModelResponse> {
    const response = this.responses[this.index++]
    if (!response) throw new Error('No scripted model response left')
    return response
  }

  async *stream(_request: ModelRequest): AsyncIterable<never> {
    yield* []
  }
}

class SubAgentGateway implements IModelGateway {
  private supervisorSpawned = false

  async complete(request: ModelRequest): Promise<ModelResponse> {
    const toolNames = request.tools?.map(t => t.name) ?? []
    if (toolNames.includes('worker-a') && toolNames.includes('worker-b')) {
      if (this.supervisorSpawned) return text('supervisor done')
      this.supervisorSpawned = true
      return tools([
        { id: 'spawn-a', name: 'worker-a', input: { goal: 'work a', input: 'do a' } },
        { id: 'spawn-b', name: 'worker-b', input: { goal: 'work b', input: 'do b' } },
      ])
    }
    return text('child done')
  }

  async *stream(_request: ModelRequest): AsyncIterable<never> {
    yield* []
  }
}

function agent(config: Partial<AgentConfig>): AgentConfig {
  return {
    agentId:      'agent',
    version:      '1.0.0',
    systemPrompt: 'scripted test agent',
    fsm:          { states: [{ name: 'react', type: 'llm', max_iterations: 5 }] },
    model:        MODEL,
    ...config,
  }
}

describe('Deterministic e2e: framework semantics', () => {
  it('executes parallel-safe tools in one LLM turn and one parallel batch', async () => {
    const searchTool: ToolDefinition = {
      name:         'search',
      description:  'scripted search',
      inputSchema:  { type: 'object', properties: { q: { type: 'string' } }, required: ['q'] },
      parallelSafe: true,
      handler:      async input => ({ ok: true, input }),
    }

    const trajectoryStore = new TrajectoryStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      trajectoryStore,
      gateway: new SequentialGateway([
        tools([
          { id: 'search-a', name: 'search', input: { q: 'a' } },
          { id: 'search-b', name: 'search', input: { q: 'b' } },
          { id: 'search-c', name: 'search', input: { q: 'c' } },
        ]),
        text('done'),
      ]),
      tools: [searchTool],
    })
    milkie.registerAgent(agent({ agentId: 'parallel-agent' }))

    const result = await milkie.invoke({ agentId: 'parallel-agent', goal: 'search', input: 'go' })
    const trajectory = await trajectoryStore.getByRunId(result.agentRunId)
    const searchSpans = trajectory.spans.filter(
      s => s.name === 'tool.call' && s.attributes['toolName'] === 'search',
    )

    expect(result.status).toBe('completed')
    expect(searchSpans).toHaveLength(3)
    expect(new Set(searchSpans.map(s => s.attributes['turn']))).toHaveProperty('size', 1)
    expect(new Set(searchSpans.map(s => s.attributes['parallelBatchId']))).toHaveProperty('size', 1)
  })

  it('spawns sub-agents concurrently with isolated child traces', async () => {
    const trajectoryStore = new TrajectoryStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      trajectoryStore,
      gateway: new SubAgentGateway(),
    })

    milkie.registerAgent(agent({ agentId: 'worker-a' }))
    milkie.registerAgent(agent({ agentId: 'worker-b' }))
    milkie.registerAgent(agent({
      agentId: 'supervisor',
      subAgents: {
        'worker-a': '1.0.0',
        'worker-b': '1.0.0',
      },
    }))

    const result = await milkie.invoke({ agentId: 'supervisor', goal: 'coordinate', input: 'start' })
    const trajectory = await trajectoryStore.getByRunId(result.agentRunId)
    const spawnSpans = trajectory.spans.filter(s => s.name === 'agent.spawn')
    const childRunSpans = trajectory.spans.filter(
      s => s.name === 'agent.run' && s.attributes['agentId'] !== 'supervisor',
    )

    expect(result.status).toBe('completed')
    expect(spawnSpans).toHaveLength(2)
    expect(spawnSpans.map(s => s.attributes['childAgentId']).sort()).toEqual(['worker-a', 'worker-b'])
    expect(new Set(spawnSpans.map(s => s.attributes['turn']))).toHaveProperty('size', 1)
    expect(new Set(childRunSpans.map(s => s.traceId))).toHaveProperty('size', 2)
  })

  it('loads skill instructions on the next context epoch after skill_request', async () => {
    const trajectoryStore = new TrajectoryStore()
    const lookupTool: ToolDefinition = {
      name:        'lookup',
      description: 'scripted lookup',
      inputSchema: { type: 'object', properties: {}, required: [] },
      handler:     async () => ({ result: 'typescript facts' }),
    }
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      trajectoryStore,
      gateway: new SequentialGateway([
        tools([{ id: 'skill-1', name: 'skill_request', input: { name: 'research' } }]),
        tools([{ id: 'lookup-1', name: 'lookup', input: {} }]),
        text('report with loaded research skill'),
      ]),
      tools: [lookupTool],
    })
    milkie.registerAgent(agent({
      agentId: 'skill-agent',
      skills: { research: '1.0.0' },
      skillInstructions: { research: 'Research skill instructions' },
    }))

    const result = await milkie.invoke({ agentId: 'skill-agent', goal: 'research', input: 'go' })
    const trajectory = await trajectoryStore.getByRunId(result.agentRunId)
    const llmSpans = trajectory.spans.filter(s => s.name === 'llm.call')

    expect(result.status).toBe('completed')
    expect(llmSpans[0]!.attributes['loadedSkills']).toEqual([])
    expect(llmSpans[1]!.attributes['loadedSkills']).toEqual(['research'])
    // contextEpoch now reflects every region mutation, not just skill loads.
    // The substrate guarantee that matters: epoch is monotonically increasing
    // across LLM calls within a run.
    const epoch0 = llmSpans[0]!.attributes['contextEpoch'] as number
    const epoch1 = llmSpans[1]!.attributes['contextEpoch'] as number
    expect(typeof epoch0).toBe('number')
    expect(epoch1).toBeGreaterThan(epoch0)
  })
})
