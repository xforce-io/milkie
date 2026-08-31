import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import {
  tryRepairJson,
  resolveControlToolCall,
  TOOL_ARGUMENTS_SCHEMA_INVALID,
  TOOL_ARGUMENTS_INVALID_JSON,
  TOOL_EXECUTION_ERROR,
} from '../runtime/toolProtocol'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { ToolCall } from '../types/tool'
import type { Plan } from '../tools/cognitive'

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

describe('#245 control tool protocol', () => {
  describe('Unit: resolveControlToolCall direct', () => {
    it('repairs trailing-comma via rawArguments', () => {
      const call: ToolCall = {
        id: 'u1',
        name: 'update_step',
        input: {},
        invalidArguments: { code: 'TOOL_ARGUMENTS_INVALID_JSON', message: 'Tool arguments are not valid JSON', rawLength: 29 },
        rawArguments: '{"stepId":0,"status":"done",}',
      }
      const resolved = resolveControlToolCall(call)
      expect(resolved.ok).toBe(true)
      if (resolved.ok) {
        expect(resolved.input).toEqual({ stepId: 0, status: 'done' })
      }
    })

    it('repairs missing closer via rawArguments', () => {
      const call: ToolCall = {
        id: 'u1',
        name: 'update_step',
        input: {},
        invalidArguments: { code: 'TOOL_ARGUMENTS_INVALID_JSON', message: 'Tool arguments are not valid JSON', rawLength: 28 },
        rawArguments: '{"stepId":1,"status":"done"',
      }
      const resolved = resolveControlToolCall(call)
      expect(resolved.ok).toBe(true)
      if (resolved.ok) {
        expect(resolved.input).toEqual({ stepId: 1, status: 'done' })
      }
    })

    it('rejects unrepairable truncation', () => {
      const call: ToolCall = {
        id: 'u1',
        name: 'update_step',
        input: {},
        invalidArguments: { code: 'TOOL_ARGUMENTS_INVALID_JSON', message: 'Tool arguments are not valid JSON', rawLength: 15 },
        rawArguments: '{"stepId":0,"st',
      }
      const resolved = resolveControlToolCall(call)
      expect(resolved.ok).toBe(false)
      if (!resolved.ok) {
        expect(resolved.code).toBe(TOOL_ARGUMENTS_INVALID_JSON)
      }
    })
  })

  describe('S1: lightweight malformed params can be repaired or get actionable feedback', () => {
    it('repairs trailing-comma update_step and accepts it', async () => {
      expect(tryRepairJson('{"stepId":0,"status":"done",}')).toEqual({ stepId: 0, status: 'done' })

      const eventStore = new MemoryEventStore()
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
        eventStore,
        gateway: gw,
      })
      milkie.registerAgent(config())
      const result = await milkie.invoke({ agentId: 'planner', goal: 'g', input: 'i' })
      expect(result.status).not.toBe('error')
      expect(result.stopReason).not.toBe('runtime_error')

      // Verify repair succeeded: handler ran and WM updated
      const events = await eventStore.readByRunId(result.agentRunId)
      const toolResponses = events.filter(e => e.type === 'tool.responded')
      const updateResponse = toolResponses.find(e => (e.payload as { toolName: string }).toolName === 'update_step')
      expect(updateResponse).toBeDefined()
      expect((updateResponse!.payload as { error?: unknown }).error).toBeUndefined()

      const wmEvents = events.filter(e => e.type === 'wm.mutated')
      expect(wmEvents.length).toBeGreaterThan(0)
      const lastWmEvent = wmEvents[wmEvents.length - 1]
      expect(lastWmEvent).toBeDefined()
      const finalWm = (lastWmEvent!.payload as { snapshot: { data: { plan?: Plan } } }).snapshot.data
      expect(finalWm.plan).toBeDefined()
      expect(finalWm.plan?.steps[0]?.status).toBe('done')
    })

    it('unrepairable truncation rejects with invalid_args and allows next turn', async () => {
      const eventStore = new MemoryEventStore()
      const truncated: ToolCall = {
        id: 'u1',
        name: 'update_step',
        input: {},
        invalidArguments: { code: 'TOOL_ARGUMENTS_INVALID_JSON', message: 'Tool arguments are not valid JSON', rawLength: 15 },
        rawArguments: '{"stepId":0,"st',
      }
      const gw = new ScriptedGateway([
        toolUse({ id: 'c1', name: 'create_plan', input: { steps: ['a'] } }),
        toolUse(truncated),
        toolUse({ id: 'u2', name: 'update_step', input: { stepId: 0, status: 'done' } }),
        { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'stop' },
      ])
      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        eventStore,
        gateway: gw,
      })
      milkie.registerAgent(config())
      const result = await milkie.invoke({ agentId: 'planner', goal: 'g', input: 'i' })
      expect(result.status).not.toBe('error')

      const events = await eventStore.readByRunId(result.agentRunId)
      const toolResponses = events.filter(e => e.type === 'tool.responded')
      expect(toolResponses.length).toBe(3) // create + bad update + good update

      const badResponse = toolResponses[1]!.payload as { toolName: string; error?: { code: string } }
      expect(badResponse.toolName).toBe('update_step')
      expect(badResponse.error).toBeDefined()
      expect(badResponse.error!.code).toBe(TOOL_ARGUMENTS_INVALID_JSON)

      // Verify handler was not called for truncated (only 2 WM events: create_plan + legal update)
      const wmEvents = events.filter(e => e.type === 'wm.mutated')
      expect(wmEvents.length).toBe(2) // create_plan and legal update_step only

      // Second legal call succeeds
      const goodResponse = toolResponses[2]!.payload as { error?: unknown }
      expect(goodResponse.error).toBeUndefined()
      
      const lastWmEvent = wmEvents[wmEvents.length - 1]
      expect(lastWmEvent).toBeDefined()
      const finalWm = (lastWmEvent!.payload as { snapshot: { data: { plan?: Plan } } }).snapshot.data
      expect(finalWm.plan?.steps[0]?.status).toBe('done')
    })

    it('missing closer is repaired when unique', async () => {
      expect(tryRepairJson('{"stepId":1,"status":"done"')).toEqual({ stepId: 1, status: 'done' })

      const eventStore = new MemoryEventStore()
      const missingClose: ToolCall = {
        id: 'u1',
        name: 'update_step',
        input: {},
        invalidArguments: { code: 'TOOL_ARGUMENTS_INVALID_JSON', message: 'Tool arguments are not valid JSON', rawLength: 28 },
        rawArguments: '{"stepId":1,"status":"done"',
      }
      const gw = new ScriptedGateway([
        toolUse({ id: 'c1', name: 'create_plan', input: { steps: ['x', 'y'] } }),
        toolUse(missingClose),
        { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'stop' },
      ])
      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        eventStore,
        gateway: gw,
      })
      milkie.registerAgent(config())
      const result = await milkie.invoke({ agentId: 'planner', goal: 'g', input: 'i' })
      expect(result.status).not.toBe('error')

      const events = await eventStore.readByRunId(result.agentRunId)
      const toolResponses = events.filter(e => e.type === 'tool.responded')
      const updateResponse = toolResponses.find(e => (e.payload as { toolName: string }).toolName === 'update_step')
      expect((updateResponse!.payload as { error?: unknown }).error).toBeUndefined()

      const wmEvents = events.filter(e => e.type === 'wm.mutated')
      const lastWmEvent = wmEvents[wmEvents.length - 1]
      expect(lastWmEvent).toBeDefined()
      const finalWm = (lastWmEvent!.payload as { snapshot: { data: { plan?: Plan } } }).snapshot.data
      expect(finalWm.plan?.steps[1]?.status).toBe('done')
    })

    it('truncated object is repaired when unique closer works', async () => {
      // Only one closer works: '}'
      const truncated = '{"stepId":0'
      const repaired = tryRepairJson(truncated)
      expect(repaired).toEqual({ stepId: 0 })
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
  })

  describe('S2: invalid_args vs tool_execution_error are distinguishable', () => {
    it('schema-invalid vs no-plan execution error have distinct codes', async () => {
      const eventStore = new MemoryEventStore()
      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        eventStore,
        gateway: new ScriptedGateway([
          toolUse({
            id: 'bad',
            name: 'update_step',
            input: { stepId: 'x', status: 'done' }, // wrong type
          }),
          toolUse({ id: 'np', name: 'update_step', input: { stepId: 0, status: 'done' } }), // no plan
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
      expect(codes[0]).not.toBe(codes[1]) // distinct
    })

    it('invalid_args path does not call handler (zero WM mutations)', async () => {
      const eventStore = new MemoryEventStore()
      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        eventStore,
        gateway: new ScriptedGateway([
          toolUse({ id: 'bad', name: 'create_plan', input: { steps: [] } }), // empty steps
          { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'stop' },
        ]),
      })
      milkie.registerAgent(config())
      const result = await milkie.invoke({ agentId: 'planner', goal: 'g', input: 'i' })
      expect(result.status).not.toBe('error')

      const events = await eventStore.readByRunId(result.agentRunId)
      const wmEvents = events.filter(e => e.type === 'wm.mutated')
      expect(wmEvents.length).toBe(0) // handler never ran

      const toolResponses = events.filter(e => e.type === 'tool.responded')
      const response = toolResponses[0]!.payload as { error?: { code: string } }
      expect(response.error).toBeDefined()
      expect(response.error!.code).toBe(TOOL_ARGUMENTS_SCHEMA_INVALID)
    })
  })

  describe('S3: plan/step use narrow input surface', () => {
    it('legal create_plan writes queryable plan', async () => {
      const eventStore = new MemoryEventStore()
      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        eventStore,
        gateway: new ScriptedGateway([
          toolUse({ id: 'c1', name: 'create_plan', input: { steps: ['one', 'two'] } }),
          { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'stop' },
        ]),
      })
      milkie.registerAgent(config())
      const result = await milkie.invoke({ agentId: 'planner', goal: 'g', input: 'i' })
      expect(result.status).toBe('completed')

      const events = await eventStore.readByRunId(result.agentRunId)
      const wmEvents = events.filter(e => e.type === 'wm.mutated')
      const wm = (wmEvents[0]!.payload as { snapshot: { data: { plan?: Plan } } }).snapshot.data
      expect(wm.plan).toBeDefined()
      expect(wm.plan!.steps).toHaveLength(2)
      expect(wm.plan?.steps[0]?.desc).toBe('one')
    })

    it('legal update_step mutates plan in WM', async () => {
      const eventStore = new MemoryEventStore()
      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        eventStore,
        gateway: new ScriptedGateway([
          toolUse({ id: 'c1', name: 'create_plan', input: { steps: ['task'] } }),
          toolUse({ id: 'u1', name: 'update_step', input: { stepId: 0, status: 'done' } }),
          { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'stop' },
        ]),
      })
      milkie.registerAgent(config())
      const result = await milkie.invoke({ agentId: 'planner', goal: 'g', input: 'i' })
      expect(result.status).toBe('completed')

      const events = await eventStore.readByRunId(result.agentRunId)
      const wmEvents = events.filter(e => e.type === 'wm.mutated')
      const lastWmEvent = wmEvents[wmEvents.length - 1]
      expect(lastWmEvent).toBeDefined()
      const finalWm = (lastWmEvent!.payload as { snapshot: { data: { plan?: Plan } } }).snapshot.data
      expect(finalWm.plan?.steps[0]?.status).toBe('done')
    })

    it('extra field returns TOOL_ARGUMENTS_SCHEMA_INVALID with field name', async () => {
      const eventStore = new MemoryEventStore()
      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        eventStore,
        gateway: new ScriptedGateway([
          toolUse({ id: 'c1', name: 'create_plan', input: { steps: ['a'], extra: 'bad' } }),
          { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'stop' },
        ]),
      })
      milkie.registerAgent(config())
      const result = await milkie.invoke({ agentId: 'planner', goal: 'g', input: 'i' })
      expect(result.status).not.toBe('error')

      const events = await eventStore.readByRunId(result.agentRunId)
      const toolResponses = events.filter(e => e.type === 'tool.responded')
      const response = toolResponses[0]!.payload as { error?: { code: string; message: string } }
      expect(response.error).toBeDefined()
      expect(response.error!.code).toBe(TOOL_ARGUMENTS_SCHEMA_INVALID)
      expect(response.error!.message).toContain('extra') // field name in message
    })

    it('bad enum returns TOOL_ARGUMENTS_SCHEMA_INVALID with hint', async () => {
      const eventStore = new MemoryEventStore()
      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        eventStore,
        gateway: new ScriptedGateway([
          toolUse({ id: 'c1', name: 'create_plan', input: { steps: ['a'] } }),
          toolUse({ id: 'u1', name: 'update_step', input: { stepId: 0, status: 'complete' } }), // wrong enum
          { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'stop' },
        ]),
      })
      milkie.registerAgent(config())
      const result = await milkie.invoke({ agentId: 'planner', goal: 'g', input: 'i' })
      expect(result.status).not.toBe('error')

      const events = await eventStore.readByRunId(result.agentRunId)
      const toolResponses = events.filter(e => e.type === 'tool.responded')
      const updateResponse = toolResponses[1]!.payload as { error?: { code: string; message: string } }
      expect(updateResponse.error).toBeDefined()
      expect(updateResponse.error!.code).toBe(TOOL_ARGUMENTS_SCHEMA_INVALID)
      expect(updateResponse.error!.message).toContain('status') // mentions the field
    })
  })
})
