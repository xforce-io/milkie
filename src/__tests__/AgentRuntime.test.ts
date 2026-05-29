import { AgentRuntime } from '../runtime/AgentRuntime'
import { DefaultIOPort } from '../runtime/IOPort'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { MemoryTraceObjectStore } from '../trace/TraceObjectStore'
import type { AgentConfig } from '../types/agent'
import type { AgentCheckpoint } from '../types/store'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { ToolDefinition } from '../types/tool'
import type { AgentReturnedPayload } from '../trace/types'

// ---- Fixtures ----

function makeConfig(overrides: Partial<AgentConfig> = {}): AgentConfig {
  return {
    agentId:      'test-agent',
    version:      '1.0.0',
    systemPrompt: 'You are a test agent.',
    fsm: {
      states: [{ name: 'react', type: 'llm' }],
    },
    model: {
      provider: 'test',
      model:    'test-model',
      adapter:  'test',
    },
    ...overrides,
  }
}

// A gateway that returns a fixed response sequence
class SequentialGateway implements IModelGateway {
  private responses: ModelResponse[]
  private index = 0

  constructor(responses: ModelResponse[]) {
    this.responses = responses
  }

  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.responses[this.index++]
    if (!r) throw new Error('No more mock responses')
    return r
  }

  async *stream(_req: ModelRequest): AsyncIterable<never> {
    yield* []
  }
}

class SupervisorGateway implements IModelGateway {
  async complete(req: ModelRequest): Promise<ModelResponse> {
    const toolNames = req.tools?.map(t => t.name) ?? []
    if (toolNames.includes('worker-a') && toolNames.includes('worker-b')) {
      return {
        content: [
          { type: 'tool_use', id: 'spawn-a', name: 'worker-a', input: { goal: 'work a', input: 'do a' } },
          { type: 'tool_use', id: 'spawn-b', name: 'worker-b', input: { goal: 'work b', input: 'do b' } },
        ],
        toolCalls: [
          { id: 'spawn-a', name: 'worker-a', input: { goal: 'work a', input: 'do a' } },
          { id: 'spawn-b', name: 'worker-b', input: { goal: 'work b', input: 'do b' } },
        ],
        finishReason: 'tool_use',
      }
    }

    await new Promise<void>(resolve => setTimeout(resolve, 100))
    return textResponse('child done')
  }

  async *stream(_req: ModelRequest): AsyncIterable<never> {
    yield* []
  }
}

function textResponse(text: string): ModelResponse {
  return {
    content:      [{ type: 'text', text }],
    toolCalls:    [],
    finishReason: 'end_turn',
  }
}

function toolCallResponse(id: string, name: string, input: unknown): ModelResponse {
  return {
    content:   [{ type: 'tool_use', id, name, input }],
    toolCalls: [{ id, name, input }],
    finishReason: 'tool_use',
  }
}

async function waitFor(
  predicate: () => Promise<boolean>,
  timeoutMs = 1000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (await predicate()) return
    await new Promise<void>(resolve => setTimeout(resolve, 10))
  }
  throw new Error('Timed out waiting for condition')
}

// ---- Tests ----

describe('AgentRuntime', () => {
  describe('single-state ReAct (type:llm)', () => {
    it('returns text output when LLM produces text with no on.DONE', async () => {
      const gateway = new SequentialGateway([textResponse('Hello, world!')])
      const recorder = new InMemoryRecorder(undefined, 'test-agent')
      const runtime = new AgentRuntime({
        config:     makeConfig(),
        goal:       'test goal',
        input:      'hi',
        stateStore: new MemoryStore(),
        recorder,
        ioPort: new DefaultIOPort(gateway),
      })

      const result = await runtime.run('hi')
      expect(result.status).toBe('completed')
      expect(result.output).toBe('Hello, world!')
    })

    it('records agent.run and llm.call spans', async () => {
      const gateway  = new SequentialGateway([textResponse('done')])
      const recorder = new InMemoryRecorder(undefined, 'test-agent')
      const runtime  = new AgentRuntime({
        config:     makeConfig(),
        goal:       'test',
        input:      'hi',
        stateStore: new MemoryStore(),
        recorder,
        ioPort: new DefaultIOPort(gateway),
      })

      await runtime.run('hi')
      const spans = recorder.getSpans()
      expect(spans.some(s => s.name === 'agent.run')).toBe(true)
      expect(spans.some(s => s.name === 'llm.call')).toBe(true)
    })

    it('executes tool call and continues loop until text output', async () => {
      const toolDef: ToolDefinition = {
        name:        'search',
        description: 'search the web',
        inputSchema: { type: 'object', properties: { q: { type: 'string' } } },
        parallelSafe: true,
        handler:     async () => ({ results: ['result1'] }),
      }

      const gateway = new SequentialGateway([
        toolCallResponse('tc-1', 'search', { q: 'test' }),
        textResponse('I found result1'),
      ])
      const recorder = new InMemoryRecorder()
      const runtime  = new AgentRuntime({
        config:     makeConfig(),
        goal:       'search something',
        input:      'search for test',
        stateStore: new MemoryStore(),
        recorder,
        ioPort: new DefaultIOPort(gateway),
        extraTools: [toolDef],
      })

      const result = await runtime.run('search for test')
      expect(result.status).toBe('completed')
      expect(result.output).toBe('I found result1')

      const toolSpans = recorder.getSpans().filter(s => s.name === 'tool.call')
      expect(toolSpans).toHaveLength(1)
      expect(toolSpans[0]?.attributes['toolName']).toBe('search')
    })
  })

  describe('multi-state FSM (intent routing)', () => {
    const routingConfig = makeConfig({
      fsm: {
        states: [
          {
            name:  'classify',
            type:  'llm',
            tools: ['classify_intent'],
            on:    { INTENT_DONE: 'done' },
          },
          { name: 'done', type: 'action', terminal: true },
        ],
      },
    })

    it('transitions via ctx.emit from tool handler', async () => {
      let emitted = false
      const classifyTool: ToolDefinition = {
        name:        'classify_intent',
        description: 'classify',
        inputSchema: { type: 'object', properties: {} },
        handler:     async (_input, ctx) => {
          emitted = true
          ctx.emit('INTENT_DONE')
          return { intent: 'done' }
        },
      }

      const gateway  = new SequentialGateway([
        toolCallResponse('tc-1', 'classify_intent', {}),
      ])
      const recorder = new InMemoryRecorder()
      const runtime  = new AgentRuntime({
        config:     routingConfig,
        goal:       'classify',
        input:      'hello',
        stateStore: new MemoryStore(),
        recorder,
        ioPort: new DefaultIOPort(gateway),
        extraTools: [classifyTool],
      })

      const result = await runtime.run('hello')
      expect(emitted).toBe(true)
      expect(result.status).toBe('completed')

      const transitions = recorder.getSpans().filter(s => s.name === 'fsm.transition')
      expect(transitions.map(t => t.attributes['toState'])).toContain('done')
    })
  })

  describe('error handling', () => {
    it('returns error status when tool throws and no recovery', async () => {
      const failingTool: ToolDefinition = {
        name:        'fail',
        description: 'always fails',
        inputSchema: { type: 'object', properties: {} },
        handler:     async () => { throw new Error('tool exploded') },
      }

      const gateway = new SequentialGateway([
        toolCallResponse('tc-1', 'fail', {}),
        textResponse('recovered'),  // LLM sees error and continues
      ])
      const recorder = new InMemoryRecorder()
      const runtime  = new AgentRuntime({
        config:     makeConfig(),
        goal:       'test',
        input:      'run fail',
        stateStore: new MemoryStore(),
        recorder,
        ioPort: new DefaultIOPort(gateway),
        extraTools: [failingTool],
      })

      // Tool fails but LLM loop continues and eventually produces text
      const result = await runtime.run('run fail')
      expect(result.status).toBe('completed')
      expect(result.output).toBe('recovered')
    })
  })

  describe('checkpoint and resume', () => {
    it('saves interrupted checkpoints as paused with a resume state', async () => {
      const stateStore = new MemoryStore()
      const runtime = new AgentRuntime({
        config:     makeConfig(),
        goal:       'test interrupt',
        input:      'hi',
        contextId:  'ctx-interrupt',
        stateStore,
        recorder:   new InMemoryRecorder('trace-interrupt', 'test-agent'),
        ioPort:     new DefaultIOPort(new SequentialGateway([textResponse('should not be called')])),
      })

      runtime.interrupt()
      const result = await runtime.run('hi')

      expect(result.status).toBe('interrupted')
      const checkpoint = await stateStore.get('context:ctx-interrupt:checkpoint:latest') as AgentCheckpoint
      expect(checkpoint.fsm.currentState).toBe('paused')
      expect(checkpoint.fsm.resumeState).toBe('react')
      expect(checkpoint.meta.contextId).toBe('ctx-interrupt')
      expect(checkpoint.meta.agentRunId).toBe(result.agentRunId)
    })

    it('resume reuses checkpoint contextId, agentRunId, and traceId', async () => {
      const stateStore = new MemoryStore()
      const checkpoint: AgentCheckpoint = {
        checkpointId: 'cp-1',
        sequence:     1,
        goal:         'resume goal',
        currentTurn:  'previous turn',
        fsm:          { currentState: 'paused', resumeState: 'react', stateData: null },
        context: {
          workingMemory: { data: {}, log: [] },
          regions:       { epoch: 0, regions: [] },
        },
        pendingEvents: [],
        children:      [],
        meta: {
          agentId:    'test-agent',
          agentRunId: 'run-original',
          timestamp:  Date.now(),
          traceId:    'trace-original',
          contextId:  'ctx-original',
        },
      }
      await stateStore.set('checkpoint-key', checkpoint)

      const milkie = new Milkie({
        stateStore,
        gateway: new SequentialGateway([textResponse('resumed')]),
      })
      milkie.registerAgent(makeConfig())

      const result = await milkie.resume('checkpoint-key', 'test-agent', 'resume goal', 'continue')

      expect(result.status).toBe('completed')
      expect(result.output).toBe('resumed')
      expect(result.agentRunId).toBe('run-original')
      expect(result.contextId).toBe('ctx-original')
    })

    it('records resume crystallization region removals after loading a checkpoint', async () => {
      const stateStore = new MemoryStore()
      const eventStore = new MemoryEventStore()
      const checkpoint: AgentCheckpoint = {
        checkpointId: 'cp-1',
        sequence:     1,
        goal:         'resume goal',
        currentTurn:  'previous turn',
        fsm:          { currentState: 'paused', resumeState: 'react', stateData: null },
        context: {
          workingMemory: { data: {}, log: [] },
          regions: {
            epoch: 1,
            regions: [{
              id:        'current-turn',
              target:    'message',
              section:   'current-turn',
              createdAt: 1,
              intraTurn: 'turn-persistent',
              interTurn: 'turn-local',
              stability: 'volatile',
              content:   'previous turn',
            } as never],
          },
        },
        pendingEvents: [],
        children:      [],
        meta: {
          agentId:    'test-agent',
          agentRunId: 'run-original',
          timestamp:  Date.now(),
          traceId:    'trace-original',
          contextId:  'ctx-original',
        },
      }
      await stateStore.set('checkpoint-key', checkpoint)

      const milkie = new Milkie({
        stateStore,
        eventStore,
        traceObjectStore: new MemoryTraceObjectStore(),
        gateway: new SequentialGateway([textResponse('resumed')]),
      })
      milkie.registerAgent(makeConfig())

      const result = await milkie.resume('checkpoint-key', 'test-agent', 'resume goal', 'continue')
      const events = await eventStore.readByRunId(result.agentRunId)

      expect(events.some(e => e.type === 'region.removed' && (e.payload as { id?: string }).id === 'current-turn')).toBe(true)
    })

    it('propagates parent interrupt to running sub-agents and records child checkpoints', async () => {
      const stateStore = new MemoryStore()
      const milkie = new Milkie({
        stateStore,
        gateway: new SupervisorGateway(),
      })

      milkie.registerAgent(makeConfig({
        agentId: 'worker-a',
        fsm: { states: [{ name: 'react', type: 'llm' }] },
      }))
      milkie.registerAgent(makeConfig({
        agentId: 'worker-b',
        fsm: { states: [{ name: 'react', type: 'llm' }] },
      }))
      milkie.registerAgent(makeConfig({
        agentId: 'supervisor',
        fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 3 }] },
        subAgents: {
          'worker-a': '1.0.0',
          'worker-b': '1.0.0',
        },
      }))

      const runPromise = milkie.invoke({
        agentId:   'supervisor',
        goal:      'coordinate workers',
        input:     'start workers',
        contextId: 'ctx-supervisor',
      })

      await waitFor(async () => {
        const children = await stateStore.get('context:ctx-supervisor:children') as Array<{ status: string }> | undefined
        return (children ?? []).filter(c => c.status === 'running').length === 2
      })

      await milkie.interrupt('ctx-supervisor')
      const result = await runPromise

      expect(result.status).toBe('interrupted')
      const parentCp = await stateStore.get('context:ctx-supervisor:checkpoint:latest') as AgentCheckpoint
      expect(parentCp.fsm.currentState).toBe('paused')
      expect(parentCp.children).toHaveLength(2)
      expect(parentCp.children.every(c => c.status === 'interrupted')).toBe(true)
      expect(parentCp.children.every(c => c.checkpointId)).toBe(true)
    })

    it('persists child runId in the parent checkpoint children records', async () => {
      const stateStore = new MemoryStore()
      const milkie = new Milkie({
        stateStore,
        gateway: new SupervisorGateway(),
      })

      milkie.registerAgent(makeConfig({
        agentId: 'worker-a',
        fsm: { states: [{ name: 'react', type: 'llm' }] },
      }))
      milkie.registerAgent(makeConfig({
        agentId: 'worker-b',
        fsm: { states: [{ name: 'react', type: 'llm' }] },
      }))
      milkie.registerAgent(makeConfig({
        agentId: 'supervisor',
        fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 3 }] },
        subAgents: {
          'worker-a': '1.0.0',
          'worker-b': '1.0.0',
        },
      }))

      const runPromise = milkie.invoke({
        agentId:   'supervisor',
        goal:      'coordinate workers',
        input:     'start workers',
        contextId: 'ctx-supervisor-runid',
      })

      await waitFor(async () => {
        const children = await stateStore.get('context:ctx-supervisor-runid:children') as Array<{ status: string }> | undefined
        return (children ?? []).filter(c => c.status === 'running').length === 2
      })

      await milkie.interrupt('ctx-supervisor-runid')
      await runPromise

      const parentCp = await stateStore.get('context:ctx-supervisor-runid:checkpoint:latest') as AgentCheckpoint
      expect(parentCp.children.length).toBeGreaterThan(0)
      for (const c of parentCp.children) {
        expect(typeof c.runId).toBe('string')
        expect((c.runId ?? '').length).toBeGreaterThan(0)
      }
    })

    it('emits agent.returned interrupted when sub-agents are interrupted', async () => {
      const stateStore = new MemoryStore()
      const eventStore = new MemoryEventStore()
      const milkie = new Milkie({
        stateStore,
        eventStore,
        gateway: new SupervisorGateway(),
      })

      milkie.registerAgent(makeConfig({
        agentId: 'worker-a',
        fsm: { states: [{ name: 'react', type: 'llm' }] },
      }))
      milkie.registerAgent(makeConfig({
        agentId: 'worker-b',
        fsm: { states: [{ name: 'react', type: 'llm' }] },
      }))
      milkie.registerAgent(makeConfig({
        agentId: 'supervisor',
        fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 3 }] },
        subAgents: {
          'worker-a': '1.0.0',
          'worker-b': '1.0.0',
        },
      }))

      const runPromise = milkie.invoke({
        agentId:   'supervisor',
        goal:      'coordinate',
        input:     'start',
        contextId: 'ctx-int',
      })

      await waitFor(async () => {
        const children = await stateStore.get('context:ctx-int:children') as Array<{ status: string }> | undefined
        return (children ?? []).some(c => c.status === 'running')
      })

      await milkie.interrupt('ctx-int')
      const result = await runPromise

      const events = await eventStore.readByRunId(result.agentRunId)
      const returned = events
        .filter(e => e.type === 'agent.returned')
        .map(e => e.payload as AgentReturnedPayload)
      expect(returned.some(p => p.status === 'interrupted')).toBe(true)
    })
  })
})
