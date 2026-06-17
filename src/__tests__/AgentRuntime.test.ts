import { AgentRuntime } from '../runtime/AgentRuntime'
import { DefaultIOPort } from '../runtime/IOPort'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { RecordingIOPort } from '../trace/RecordingIOPort'
import { checkpointFromEvents } from '../trace/diagnostics/checkpointFromEvents'
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
      const eventStore = new MemoryEventStore()
      const runtime = new AgentRuntime({
        config:     makeConfig(),
        goal:       'test interrupt',
        input:      'hi',
        contextId:  'ctx-interrupt',
        stateStore,
        eventStore,
        recorder:   new InMemoryRecorder('trace-interrupt', 'test-agent'),
        ioPort:     new DefaultIOPort(new SequentialGateway([textResponse('should not be called')])),
      })

      runtime.interrupt()
      const result = await runtime.run('hi')

      expect(result.status).toBe('interrupted')
      // #73: resume state lives in the event log (agent.checkpoint event).
      const checkpoint = checkpointFromEvents(await eventStore.readByRunId(result.agentRunId))!
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
      const parentCp = checkpointFromEvents(await eventStore.readByRunId(result.agentRunId))!
      expect(parentCp.fsm.currentState).toBe('paused')
      expect(parentCp.children).toHaveLength(2)
      expect(parentCp.children.every(c => c.status === 'interrupted')).toBe(true)
      expect(parentCp.children.every(c => c.checkpointId)).toBe(true)
    })

    it('persists child runId in the parent checkpoint children records', async () => {
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
        goal:      'coordinate workers',
        input:     'start workers',
        contextId: 'ctx-supervisor-runid',
      })

      await waitFor(async () => {
        const children = await stateStore.get('context:ctx-supervisor-runid:children') as Array<{ status: string }> | undefined
        return (children ?? []).filter(c => c.status === 'running').length === 2
      })

      await milkie.interrupt('ctx-supervisor-runid')
      const result = await runPromise

      const parentCp = checkpointFromEvents(await eventStore.readByRunId(result.agentRunId))!
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

describe('#31 guard evaluation capture', () => {
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

  it('writes guardEvaluations onto the fsm.transition event', async () => {
    const eventStore = new MemoryEventStore()
    const guardTool: ToolDefinition = {
      name:        'classify_intent',
      description: 'classify',
      inputSchema: { type: 'object', properties: {} },
      handler:     async (_input, ctx) => {
        ctx.emit('INTENT_DONE', undefined, {
          guardId: 'intent-threshold', result: 'INTENT_DONE',
          contextSlice: { confidence: 0.9, threshold: 0.75 },
        })
        return { ok: true }
      },
    }
    const runtime = new AgentRuntime({
      config:     routingConfig,
      goal:       'classify',
      input:      'hello',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(new SequentialGateway([
        toolCallResponse('tc-1', 'classify_intent', {}),
      ])),
      extraTools: [guardTool],
      eventStore,
    })

    const result = await runtime.run('hello')
    const events = await eventStore.readByRunId(result.agentRunId)
    const transitions = events.filter(e => e.type === 'fsm.transition')
    const withGuard = transitions.find(
      t => (t.payload as import('../trace/types').FsmTransitionPayload).guardEvaluations,
    )
    expect((withGuard!.payload as import('../trace/types').FsmTransitionPayload).guardEvaluations)
      .toEqual([{ guardId: 'intent-threshold', result: 'INTENT_DONE', contextSlice: { confidence: 0.9, threshold: 0.75 } }])
  })

  it('omits guardEvaluations when the tool does not report one', async () => {
    const eventStore = new MemoryEventStore()
    const plainTool: ToolDefinition = {
      name:        'classify_intent',
      description: 'classify',
      inputSchema: { type: 'object', properties: {} },
      handler:     async (_input, ctx) => { ctx.emit('INTENT_DONE'); return {} },
    }
    const runtime = new AgentRuntime({
      config:     routingConfig,
      goal:       'classify',
      input:      'hello',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(new SequentialGateway([
        toolCallResponse('tc-1', 'classify_intent', {}),
      ])),
      extraTools: [plainTool],
      eventStore,
    })
    const result = await runtime.run('hello')
    const events = await eventStore.readByRunId(result.agentRunId)
    const transition = events.find(
      e => e.type === 'fsm.transition'
        && (e.payload as import('../trace/types').FsmTransitionPayload).trigger.name === 'INTENT_DONE',
    )
    expect((transition!.payload as import('../trace/types').FsmTransitionPayload).guardEvaluations)
      .toBeUndefined()
  })
})

describe('#82 per-turn variables', () => {
  class CaptureGateway implements IModelGateway {
    captured: ModelRequest[] = []
    async complete(req: ModelRequest): Promise<ModelResponse> {
      this.captured.push(req)
      return textResponse('done')
    }
    async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
  }

  it('injects variables into the messages, never into the system block', async () => {
    const gw = new CaptureGateway()
    const runtime = new AgentRuntime({
      config:     makeConfig(),
      goal:       'g',
      input:      'hi',
      variables:  { current_time: '2026-06-01T00:00:00Z', workspace: 'demo' },
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(gw),
    })

    await runtime.run('hi')

    const req = gw.captured[0]!
    const msgText = req.messages
      .flatMap(m => m.content)
      .map(c => (c.type === 'text' ? c.text : ''))
      .join('\n')
    expect(msgText).toContain('current_time')
    expect(msgText).toContain('2026-06-01T00:00:00Z')
    expect(msgText).toContain('demo')
    // prefix-cache safety: per-turn variables must never enter the system prefix
    expect(req.system ?? '').not.toContain('current_time')
    expect(req.system ?? '').not.toContain('demo')
  })

  it('keeps the system block byte-identical when only variables differ', async () => {
    const systemFor = async (variables: Record<string, string>): Promise<string> => {
      const gw = new CaptureGateway()
      await new AgentRuntime({
        config:     makeConfig(),
        goal:       'g',
        input:      'hi',
        variables,
        stateStore: new MemoryStore(),
        recorder:   new InMemoryRecorder(),
        ioPort:     new DefaultIOPort(gw),
      }).run('hi')
      return gw.captured[0]!.system ?? ''
    }

    const s1 = await systemFor({ current_time: 'T1' })
    const s2 = await systemFor({ current_time: 'T2' })
    expect(s1).toBe(s2)
  })

  it('adds no turn-context message when no variables are supplied', async () => {
    const gw = new CaptureGateway()
    await new AgentRuntime({
      config:     makeConfig(),
      goal:       'g',
      input:      'hi',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(gw),
    }).run('hi')

    const req = gw.captured[0]!
    const msgText = req.messages
      .flatMap(m => m.content)
      .map(c => (c.type === 'text' ? c.text : ''))
      .join('\n')
    expect(msgText).not.toContain('--- Turn Context ---')
  })

  it('Milkie.invoke forwards request.variables into the turn', async () => {
    const gw = new CaptureGateway()
    const milkie = new Milkie({ stateStore: new MemoryStore(), gateway: gw })
    milkie.registerAgent(makeConfig({
      agentId: 'var-agent',
      fsm: { states: [{ name: 'react', type: 'llm' }] },
    }))

    await milkie.invoke({
      agentId:   'var-agent',
      goal:      'g',
      input:     'hi',
      variables: { session_id: 'sess-9', foo: 'BAR' },
    })

    const req = gw.captured[0]!
    const msgText = req.messages
      .flatMap(m => m.content)
      .map(c => (c.type === 'text' ? c.text : ''))
      .join('\n')
    expect(msgText).toContain('session_id')
    expect(msgText).toContain('sess-9')
    expect(msgText).toContain('BAR')
  })
})

describe('#81 readable tool payload through a run', () => {
  it('stamps the LLM tool_use id onto tool.requested/responded so they pair', async () => {
    const eventStore = new MemoryEventStore()
    const toolDef: ToolDefinition = {
      name:        'search',
      description: 'search the web',
      inputSchema: { type: 'object', properties: { q: { type: 'string' } } },
      handler:     async () => ({ results: ['result1'] }),
    }
    const runId  = 'run-81'
    const ioPort = new RecordingIOPort(
      new DefaultIOPort(new SequentialGateway([
        toolCallResponse('tc-xyz', 'search', { q: 'test' }),
        textResponse('done'),
      ])),
      eventStore,
      runId,
    )
    const runtime = new AgentRuntime({
      config:     makeConfig(),
      goal:       'search something',
      input:      'go',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort,
      extraTools: [toolDef],
      eventStore,
      agentRunId: runId,
    })

    const result = await runtime.run('go')
    const events = await eventStore.readByRunId(result.agentRunId)
    const req = events.find(e => e.type === 'tool.requested')!.payload as import('../trace/types').ToolRequestedPayload
    const res = events.find(e => e.type === 'tool.responded')!.payload as import('../trace/types').ToolRespondedPayload

    expect(req.toolCallId).toBe('tc-xyz')
    expect(res.toolCallId).toBe('tc-xyz')
    expect(res.toolCallId).toBe(req.toolCallId)
    expect(res.status).toBe('ok')
    expect(res.output).toEqual({ results: ['result1'] })
  })
})

describe('#83 session-context variables', () => {
  class CaptureGateway implements IModelGateway {
    captured: ModelRequest[] = []
    async complete(req: ModelRequest): Promise<ModelResponse> {
      this.captured.push(req)
      return textResponse('done')
    }
    async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
  }

  const msgText = (req: ModelRequest): string =>
    req.messages.flatMap(m => m.content).map(c => (c.type === 'text' ? c.text : '')).join('\n')

  const sectionText = (req: ModelRequest, marker: string): string => {
    const m = req.messages.find(m => m.content.some(c => c.type === 'text' && c.text.includes(marker)))
    return m ? m.content.map(c => (c.type === 'text' ? c.text : '')).join('') : ''
  }

  it('injects sessionVariables into messages, never into the system block', async () => {
    const gw = new CaptureGateway()
    await new AgentRuntime({
      config:     makeConfig(),
      goal:       'g',
      input:      'hi',
      sessionVariables: { workspace_instructions: '用中文', session_id: 's-9' },
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(gw),
    }).run('hi')

    const req = gw.captured[0]!
    expect(msgText(req)).toContain('Session Context')
    expect(msgText(req)).toContain('workspace_instructions')
    expect(msgText(req)).toContain('用中文')
    expect(msgText(req)).toContain('session_id')
    // history-cache safety: session vars must never enter the system prefix
    expect(req.system ?? '').not.toContain('workspace_instructions')
    expect(req.system ?? '').not.toContain('用中文')
  })

  it('turn variables override same-named session vars (O2), rendered once', async () => {
    const gw = new CaptureGateway()
    await new AgentRuntime({
      config:     makeConfig(),
      goal:       'g',
      input:      'hi',
      sessionVariables: { workspace_instructions: 'OLD', session_id: 's-9' },
      variables:        { workspace_instructions: 'NEW', current_time: 'T1' },
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(gw),
    }).run('hi')

    const req = gw.captured[0]!
    const session = sectionText(req, 'Session Context')
    const turn    = sectionText(req, 'Turn Context')

    // session-context keeps un-overridden keys, drops the overridden one
    expect(session).toContain('session_id')
    expect(session).not.toContain('OLD')
    expect(session).not.toContain('workspace_instructions')
    // turn-context carries the override
    expect(turn).toContain('NEW')
    // overall the overridden key renders exactly once (in turn-context)
    expect((msgText(req).match(/workspace_instructions/g) ?? []).length).toBe(1)
  })

  it('adds no session-context message when no sessionVariables supplied', async () => {
    const gw = new CaptureGateway()
    await new AgentRuntime({
      config:     makeConfig(),
      goal:       'g',
      input:      'hi',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(gw),
    }).run('hi')

    expect(msgText(gw.captured[0]!)).not.toContain('Session Context')
  })

  it('Milkie.invoke reads stored context vars and makes them visible to the agent', async () => {
    const gw = new CaptureGateway()
    const milkie = new Milkie({ stateStore: new MemoryStore(), gateway: gw })
    milkie.registerAgent(makeConfig({
      agentId: 'ctx-agent',
      fsm: { states: [{ name: 'react', type: 'llm' }] },
    }))

    // a background writer stores a var out-of-band, before any invoke
    await milkie.setContextVar('ctx-1', 'workspace_instructions', '用中文')

    await milkie.invoke({ agentId: 'ctx-agent', goal: 'g', input: 'hi', contextId: 'ctx-1' })

    const req = gw.captured[0]!
    expect(msgText(req)).toContain('Session Context')
    expect(msgText(req)).toContain('workspace_instructions')
    expect(msgText(req)).toContain('用中文')
    expect(req.system ?? '').not.toContain('workspace_instructions')
  })
})

// #148 e2e:agent 经 run_command 取证 → 该 stdout 铸的 shell:stdout 对象可被 cite
// resolve(端到端:run_command 铸对象 → resolveObject → cite 记 cites 关系)。
describe('#148 run_command output is citable end-to-end', () => {
  class CiteRunCommandGateway implements IModelGateway {
    private cited = false
    async complete(req: ModelRequest): Promise<ModelResponse> {
      const blob = JSON.stringify(req)
      const m = blob.match(/(obj:sha256:[0-9a-f]+)/)  // run_command 铸的 objectId(tool_result 内被转义,直接配值)
      if (!m) return toolCallResponse('tc-run', 'run_command', { command: 'echo EVIDENCE-148' })
      if (!this.cited) {
        this.cited = true
        return toolCallResponse('tc-cite', 'cite', { claim: 'evidence is 148', objectId: m[1] })
      }
      return textResponse('done')
    }
    async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
  }

  it('run_command stdout objectId resolves through cite → records a cites relation', async () => {
    const stateStore = new MemoryStore()
    const eventStore = new MemoryEventStore()
    const milkie = new Milkie({
      stateStore,
      eventStore,
      traceObjectStore: new MemoryTraceObjectStore(),
      gateway: new CiteRunCommandGateway(),
    })
    milkie.registerAgent(makeConfig({ fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 5 }] } }))

    const result = await milkie.invoke({
      agentId: 'test-agent', goal: 'verify', input: 'cite the fetched evidence', contextId: 'ctx-cite-148',
    })
    const events = await eventStore.readByRunId(result.agentRunId)

    // cite 成功 = 记录了一条 cites 关系(说明 run_command 铸的 objectId 被 resolve)
    const rels = events.filter(e => e.type === 'relation.created').map(e => e.payload as { type?: string })
    expect(rels.some(p => p.type === 'cites')).toBe(true)
    // 且确有 shell:stdout 对象被 promote(object.created)
    const objs = events.filter(e => e.type === 'object.created').map(e => e.payload as { type?: string })
    expect(objs.some(p => p.type === 'shell:stdout')).toBe(true)
  })

  // #175 切片 1.2a：RunLifecycle 成为 run 最终状态的权威（保行为）。
  describe('RunLifecycle authority (#175)', () => {
    it('exposes lifecycle "completed" after a successful run', async () => {
      const runtime = new AgentRuntime({
        config:     makeConfig(),
        goal:       'test', input: 'hi',
        stateStore: new MemoryStore(),
        recorder:   new InMemoryRecorder(undefined, 'test-agent'),
        ioPort:     new DefaultIOPort(new SequentialGateway([textResponse('done')])),
      })
      const result = await runtime.run('hi')
      expect(result.status).toBe('completed')
      expect(runtime.lifecycleState).toBe('completed')
    })

    it('exposes lifecycle "interrupted" when interrupted before completing', async () => {
      const runtime = new AgentRuntime({
        config:     makeConfig(),
        goal:       'test', input: 'hi',
        stateStore: new MemoryStore(),
        recorder:   new InMemoryRecorder(undefined, 'test-agent'),
        ioPort:     new DefaultIOPort(new SequentialGateway([textResponse('nope')])),
      })
      runtime.interrupt()
      const result = await runtime.run('hi')
      expect(result.status).toBe('interrupted')
      expect(runtime.lifecycleState).toBe('interrupted')
    })

    it('exposes lifecycle "failed" when the run errors', async () => {
      const runtime = new AgentRuntime({
        config:     makeConfig(),
        goal:       'test', input: 'hi',
        stateStore: new MemoryStore(),
        recorder:   new InMemoryRecorder(undefined, 'test-agent'),
        ioPort:     new DefaultIOPort(new SequentialGateway([])),  // no responses → LLM call throws
      })
      const result = await runtime.run('hi')
      expect(result.status).toBe('error')
      expect(runtime.lifecycleState).toBe('failed')
    })

    it('persists lifecycle "interrupted" into the checkpoint for resume', async () => {
      const eventStore = new MemoryEventStore()
      const runtime = new AgentRuntime({
        config:     makeConfig(),
        goal:       'test', input: 'hi', contextId: 'ctx-lc',
        stateStore: new MemoryStore(),
        eventStore,
        recorder:   new InMemoryRecorder('trace-lc', 'test-agent'),
        ioPort:     new DefaultIOPort(new SequentialGateway([textResponse('nope')])),
      })
      runtime.interrupt()
      const result = await runtime.run('hi')
      expect(result.status).toBe('interrupted')

      const checkpoint = checkpointFromEvents(await eventStore.readByRunId(result.agentRunId))!
      expect(checkpoint.lifecycle?.state).toBe('interrupted')
      // D7: old fsm field stays readable alongside.
      expect(checkpoint.fsm.currentState).toBe('paused')
      expect(checkpoint.fsm.resumeState).toBe('react')
    })
  })
})
