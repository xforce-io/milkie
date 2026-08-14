import { Milkie } from '../../src/runtime/Milkie.js'
import { MemoryStore } from '../../src/store/MemoryStore.js'
import { MemoryEventStore } from '../../src/trace/MemoryEventStore.js'
import type { AgentConfig } from '../../src/types/agent.js'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model.js'
import type { ToolCall } from '../../src/types/tool.js'
import type { Plan } from '../../src/tools/cognitive.js'

describe('s-018: control tool protocol (stub LLM)', () => {
  function config(): AgentConfig {
    return {
      agentId: 'planner',
      version: '1',
      systemPrompt: 'You are a task planner.',
      fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 10 }] },
      model: { provider: 'stub', model: 'stub', adapter: 'stub' },
    }
  }

  class StubGateway implements IModelGateway {
    private i = 0
    constructor(private readonly replies: ModelResponse[]) {}

    async complete(_req: ModelRequest): Promise<ModelResponse> {
      const r = this.replies[this.i++]
      if (!r) return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'stop' }
      return r
    }

    async *stream(): AsyncIterable<never> {
      yield* []
    }
  }

  function toolUse(call: ToolCall): ModelResponse {
    return {
      content: [{
        type: 'tool_use',
        id: call.id,
        name: call.name,
        input: call.input,
        ...(call.invalidArguments ? { invalidArguments: call.invalidArguments } : {}),
      }],
      toolCalls: [{
        ...call,
        ...(call.rawArguments ? { rawArguments: call.rawArguments } : {}),
      }],
      finishReason: 'tool_use',
    }
  }

  it('stub create_plan → trailing-comma update_step → legal update_step → text', async () => {
    const gateway = new StubGateway([
      // 1. Legal create_plan
      toolUse({
        id: 'c1',
        name: 'create_plan',
        input: { steps: ['task-one', 'task-two'] },
      }),
      // 2. Malformed update_step with trailing comma (repairable)
      toolUse({
        id: 'u1',
        name: 'update_step',
        input: {},
        invalidArguments: {
          code: 'TOOL_ARGUMENTS_INVALID_JSON',
          message: 'Tool arguments are not valid JSON',
          rawLength: 29,
        },
        rawArguments: '{"stepId":0,"status":"done",}',
      }),
      // 3. Legal update_step
      toolUse({
        id: 'u2',
        name: 'update_step',
        input: { stepId: 1, status: 'done' },
      }),
      // 4. Text finish
      {
        content: [{ type: 'text', text: 'All tasks completed!' }],
        toolCalls: [],
        finishReason: 'stop',
      },
    ])

    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway,
    })

    milkie.registerAgent(config())
    const result = await milkie.invoke({ agentId: 'planner', goal: 'Complete tasks', input: 'Do task-one and task-two' })

    // Run should succeed
    expect(result.status).toBe('completed')
    expect(result.stopReason).toBe('model_stop')

    const events = await eventStore.readByRunId(result.agentRunId)
    const toolResponses = events.filter(e => e.type === 'tool.responded')

    // All 3 tool calls should succeed (including the repaired one)
    expect(toolResponses).toHaveLength(3)
    
    const createPlanResponse = toolResponses.find(e => (e.payload as { toolName: string }).toolName === 'create_plan')
    expect(createPlanResponse).toBeDefined()
    expect((createPlanResponse!.payload as { error?: unknown }).error).toBeUndefined()

    const updateResponses = toolResponses.filter(e => (e.payload as { toolName: string }).toolName === 'update_step')
    expect(updateResponses).toHaveLength(2)
    
    // Both update_step calls should succeed (first one repaired)
    updateResponses.forEach(r => {
      expect((r.payload as { error?: unknown }).error).toBeUndefined()
    })

    // WM should have both steps marked as done
    const wmEvents = events.filter(e => e.type === 'wm.mutated')
    expect(wmEvents.length).toBeGreaterThanOrEqual(3) // create_plan + 2x update_step
    
    const finalWm = (wmEvents[wmEvents.length - 1]!.payload as { snapshot: { data: { plan?: Plan } } }).snapshot.data
    expect(finalWm.plan).toBeDefined()
    expect(finalWm.plan!.steps).toHaveLength(2)
    expect(finalWm.plan!.steps[0]!.status).toBe('done')
    expect(finalWm.plan!.steps[1]!.status).toBe('done')
  })

  it('unrepairable JSON rejects without failing the run', async () => {
    const gateway = new StubGateway([
      toolUse({
        id: 'c1',
        name: 'create_plan',
        input: { steps: ['task'] },
      }),
      // Truncated JSON (unrepairable)
      toolUse({
        id: 'u1',
        name: 'update_step',
        input: {},
        invalidArguments: {
          code: 'TOOL_ARGUMENTS_INVALID_JSON',
          message: 'Tool arguments are not valid JSON',
          rawLength: 15,
        },
        rawArguments: '{"stepId":0,"st',
      }),
      // Model corrects and sends legal call
      toolUse({
        id: 'u2',
        name: 'update_step',
        input: { stepId: 0, status: 'done' },
      }),
      {
        content: [{ type: 'text', text: 'Fixed and done' }],
        toolCalls: [],
        finishReason: 'stop',
      },
    ])

    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway,
    })

    milkie.registerAgent(config())
    const result = await milkie.invoke({ agentId: 'planner', goal: 'task', input: 'Do it' })

    // Run should not fail
    expect(result.status).not.toBe('error')
    expect(result.stopReason).toBe('model_stop')

    const events = await eventStore.readByRunId(result.agentRunId)
    const toolResponses = events.filter(e => e.type === 'tool.responded')
    expect(toolResponses).toHaveLength(3)

    // First update should have error
    const badUpdate = toolResponses[1]!.payload as { toolName: string; error?: { code: string } }
    expect(badUpdate.toolName).toBe('update_step')
    expect(badUpdate.error).toBeDefined()
    expect(badUpdate.error!.code).toBe('TOOL_ARGUMENTS_INVALID_JSON')

    // Second update should succeed
    const goodUpdate = toolResponses[2]!.payload as { toolName: string; error?: unknown }
    expect(goodUpdate.toolName).toBe('update_step')
    expect(goodUpdate.error).toBeUndefined()

    // Final WM should have the step done
    const wmEvents = events.filter(e => e.type === 'wm.mutated')
    const finalWm = (wmEvents[wmEvents.length - 1]!.payload as { snapshot: { data: { plan?: Plan } } }).snapshot.data
    expect(finalWm.plan!.steps[0]!.status).toBe('done')
  })
})
