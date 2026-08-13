import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { tryRepairJson, TOOL_ARGUMENTS_SCHEMA_INVALID, TOOL_EXECUTION_ERROR } from '../runtime/toolProtocol'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { ToolCall } from '../types/tool'

function config(): AgentConfig {
  return {
    agentId: 'planner',
    version: '1',
    systemPrompt: 's',
    fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 6 }] },
    model: { provider: 't', model: 't', adapter: 't' },
  }
}

class ScriptedGateway implements IModelGateway {
  private i = 0
  constructor(private readonly replies: ModelResponse[]) {}
  handlerHits = { create_plan: 0, update_step: 0 }
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.replies[this.i++]
    if (!r) return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'stop' }
    return r
  }
  async *stream(): AsyncIterable<never> { yield* [] }
}

function toolUse(call: ToolCall): ModelResponse {
  return {
    content: [{ type: 'tool_use', id: call.id, name: call.name, input: call.input, ...(call.invalidArguments ? { invalidArguments: call.invalidArguments } : {}) }],
    toolCalls: [call],
    finishReason: 'tool_use',
  }
}

describe('#245 control tool protocol', () => {
  it('repairs trailing-comma update_step without calling handler on the bad parse', async () => {
    expect(tryRepairJson('{"stepId":0,"status":"done",}')).toEqual({ stepId: 0, status: 'done' })

    const bad: ToolCall = {
      id: 'u1',
      name: 'update_step',
      input: {},
      invalidArguments: { code: 'TOOL_ARGUMENTS_INVALID_JSON', message: 'Tool arguments are not valid JSON', rawLength: 29 },
      rawArguments: '{"stepId":0,"status":"done",}',
    }
    const gw = new ScriptedGateway([
      toolUse({ id: 'c1', name: 'create_plan', input: { steps: ['a', 'b'] } }),
      toolUse(bad),
      { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'stop' },
    ])
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: gw,
    })
    milkie.registerAgent(config())
    const result = await milkie.invoke({ agentId: 'planner', goal: 'g', input: 'i' })
    expect(result.status).not.toBe('error')
    expect(result.stopReason).not.toBe('runtime_error')
  })

  it('schema-invalid create_plan does not fail the run', async () => {
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: new ScriptedGateway([
        toolUse({ id: 'c1', name: 'create_plan', input: { steps: 'nope' } }),
        { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'stop' },
      ]),
    })
    milkie.registerAgent(config())
    const result = await milkie.invoke({ agentId: 'planner', goal: 'g', input: 'i' })
    expect(result.status).not.toBe('error')
  })

  it('S2: invalid args vs no-plan execution error are distinct', async () => {
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      gateway: new ScriptedGateway([
        toolUse({
          id: 'bad',
          name: 'update_step',
          input: { stepId: 'x', status: 'done' },
        }),
        toolUse({ id: 'np', name: 'update_step', input: { stepId: 0, status: 'done' } }),
        { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'stop' },
      ]),
    })
    milkie.registerAgent(config())
    const result = await milkie.invoke({ agentId: 'planner', goal: 'g', input: 'i' })
    expect(result.status).not.toBe('error')
    const responses = (await eventStore.readByRunId(result.agentRunId))
      .filter(e => e.type === 'tool.responded')
      .map(e => e.payload as { toolName: string; error?: { code?: string } | string })
    const codes = responses.map(p => typeof p.error === 'object' ? p.error.code : undefined)
    expect(codes).toContain(TOOL_ARGUMENTS_SCHEMA_INVALID)
    expect(codes).toContain(TOOL_EXECUTION_ERROR)
  })

  it('S3: legal create_plan then update_step writes plan', async () => {
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway: new ScriptedGateway([
        toolUse({ id: 'c1', name: 'create_plan', input: { steps: ['one', 'two'] } }),
        toolUse({ id: 'u1', name: 'update_step', input: { stepId: 0, status: 'done' } }),
        { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'stop' },
      ]),
    })
    milkie.registerAgent(config())
    const result = await milkie.invoke({ agentId: 'planner', goal: 'g', input: 'i' })
    expect(result.status).toBe('completed')
    expect(result.stopReason).toBe('model_stop')
  })
})
