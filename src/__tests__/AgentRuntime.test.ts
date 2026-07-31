import { AgentRuntime } from '../runtime/AgentRuntime'
import { DefaultIOPort } from '../runtime/IOPort'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { WorkingMemory } from '../store/WorkingMemory'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { RecordingIOPort } from '../trace/RecordingIOPort'
import { checkpointFromEvents } from '../trace/diagnostics/checkpointFromEvents'
import { MemoryTraceObjectStore } from '../trace/TraceObjectStore'
import type { AgentConfig } from '../types/agent'
import type { AgentCheckpoint } from '../types/store'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { ToolDefinition } from '../types/tool'
import type { AgentReturnedPayload, ToolRequestedPayload, ToolRespondedPayload } from '../trace/types'

import fs from 'fs'
import os from 'os'
import path from 'path'
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

  requests: ModelRequest[] = []

  async complete(req: ModelRequest): Promise<ModelResponse> {
    this.requests.push(req)
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

function loadAgentFromFrontmatter(frontmatter: string): AgentConfig {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-agent-config-'))
  const agentFile = path.join(tmpDir, 'agent.md')
  fs.writeFileSync(agentFile, `---\n${frontmatter}\n---\nYou are a test agent.`)

  try {
    return new Milkie({ stateStore: new MemoryStore() }).loadAgentFile(agentFile)
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true })
  }
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
      // #175 §8: v2 checkpoint — lifecycle, no fsm.
      const checkpoint = checkpointFromEvents(await eventStore.readByRunId(result.agentRunId))!
      expect(checkpoint.schemaVersion).toBe(2)
      expect(checkpoint.lifecycle?.status).toBe('interrupted')
      expect(checkpoint.lifecycle?.resumeKind).toBe('loop')
      expect(checkpoint.fsm).toBeUndefined()
      // #181: the #60 pendingEvents residue is gone — never written to checkpoints.
      expect(checkpoint.pendingEvents).toBeUndefined()
      expect(checkpoint.meta.contextId).toBe('ctx-interrupt')
      expect(checkpoint.meta.agentRunId).toBe(result.agentRunId)
    })

    it('#172: persists a checkpoint with the WM written in a user-defined terminal turn', async () => {
      const stateStore = new MemoryStore()
      const eventStore = new MemoryEventStore()
      const rememberTool: ToolDefinition = {
        name:        'remember',
        description: 'write to working memory',
        inputSchema: { type: 'object', properties: { value: { type: 'string' } } },
        handler:     async (input, ctx) => {
          ctx.workingMemory.set('note', (input as { value: string }).value)
          return { ok: true }
        },
      }
      const runtime = new AgentRuntime({
        config: makeConfig({
          fsm: { states: [{ name: 'completed', type: 'llm', terminal: true }] },
        }),
        goal:       'terminal turn',
        input:      'go',
        contextId:  'ctx-terminal',
        stateStore,
        eventStore,
        recorder:   new InMemoryRecorder('trace-terminal', 'test-agent'),
        ioPort:     new DefaultIOPort(new SequentialGateway([
          toolCallResponse('tc-1', 'remember', { value: 'kept' }),
          textResponse('all done'),
        ])),
        extraTools: [rememberTool],
      })

      const result = await runtime.run('go')
      expect(result.status).toBe('completed')

      // The recovered checkpoint contains the WM written in the terminal turn.
      // Read it back through the public WorkingMemory API rather than reaching
      // into the serialised internals, so the test tracks the contract (the WM
      // value is recoverable) and not the storage shape.
      const checkpoint = checkpointFromEvents(await eventStore.readByRunId(result.agentRunId))!
      expect(checkpoint).toBeDefined()
      const recoveredWM = WorkingMemory.fromJSON(checkpoint.context.workingMemory)
      expect(recoveredWM.get('note')).toBe('kept')
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
      expect(parentCp.lifecycle?.status).toBe('interrupted')
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

describe('#219 invalid tool arguments', () => {
  it('rejects malformed JSON before the handler, records its identity, and still permits a valid empty input', async () => {
    const eventStore = new MemoryEventStore()
    const handler = jest.fn(async () => ({ ok: true }))
    const gateway = new SequentialGateway([
      {
        content: [{
          type: 'tool_use',
          id:   'invalid-call',
          name: 'search',
          input: {},
          invalidArguments: {
            code:      'TOOL_ARGUMENTS_INVALID_JSON',
            message:   'Tool arguments are not valid JSON',
            rawLength: 9,
          },
        }],
        toolCalls: [{
          id:   'invalid-call',
          name: 'search',
          input: {},
          invalidArguments: {
            code:      'TOOL_ARGUMENTS_INVALID_JSON',
            message:   'Tool arguments are not valid JSON',
            rawLength: 9,
          },
        }],
        finishReason: 'tool_use',
      },
      toolCallResponse('valid-empty-call', 'search', {}),
      textResponse('done'),
    ])
    const runId = 'run-219'
    const runtime = new AgentRuntime({
      config:     makeConfig(),
      goal:       'test malformed tool arguments',
      input:      'go',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new RecordingIOPort(new DefaultIOPort(gateway), eventStore, runId),
      eventStore,
      agentRunId: runId,
      extraTools: [{
        name:        'search',
        description: 'search',
        inputSchema: { type: 'object', properties: {} },
        handler,
      }],
    })

    await expect(runtime.run('go')).resolves.toMatchObject({ status: 'completed', output: 'done' })
    expect(handler).toHaveBeenCalledTimes(1)
    expect(handler).toHaveBeenCalledWith({}, expect.anything())

    const rejectedResult = gateway.requests[1]!.messages
      .flatMap(message => message.content)
      .find(content => content.type === 'tool_result')
    expect(rejectedResult).toMatchObject({
      is_error: true,
      content: JSON.stringify({
        code:    'TOOL_ARGUMENTS_INVALID_JSON',
        message: 'Tool arguments are not valid JSON',
      }),
    })

    const events = await eventStore.readByRunId(runId)
    const requests = events
      .filter(event => event.type === 'tool.requested')
      .map(event => event.payload as ToolRequestedPayload)
    const responses = events
      .filter(event => event.type === 'tool.responded')
      .map(event => event.payload as ToolRespondedPayload)
    expect(requests).toHaveLength(2)
    expect(requests[0]).toMatchObject({
      toolCallId: 'invalid-call',
      invalidArguments: { code: 'TOOL_ARGUMENTS_INVALID_JSON' },
    })
    expect(responses[0]).toMatchObject({
      toolCallId: 'invalid-call',
      status:     'error',
      error:      { code: 'TOOL_ARGUMENTS_INVALID_JSON' },
    })
    expect(requests[0]!.requestHash).not.toBe(requests[1]!.requestHash)
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
    it('returns and persists a structured error when max iterations are exhausted', async () => {
      const stateStore = new MemoryStore()
      const eventStore = new MemoryEventStore()
      const milkie = new Milkie({
        stateStore,
        eventStore,
        gateway: new SequentialGateway([]),
      })
      milkie.registerAgent(makeConfig({
        fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 0 }] },
      }))

      const result = await milkie.invoke({
        agentId: 'test-agent', goal: 'stop', input: 'stop', contextId: 'ctx-max-iterations',
      })

      expect(result).toMatchObject({
        status: 'error',
        error: {
          code: 'MAX_ITERATIONS_EXCEEDED',
          phase: 'agent_loop',
          retryable: false,
          message: 'State "react" exceeded max_iterations (0)',
        },
      })
      const terminal = (await eventStore.readByRunId(result.agentRunId))
        .find(event => event.type === 'agent.run.completed')
      expect(terminal?.payload).toMatchObject({ status: 'error', error: result.error })
    })

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
      // #175 §8: v2 — explicit schemaVersion + lifecycle, no fsm written.
      expect(checkpoint.schemaVersion).toBe(2)
      expect(checkpoint.lifecycle?.status).toBe('interrupted')
      expect(checkpoint.fsm).toBeUndefined()
    })
  })
})

describe('agent frontmatter fsm configuration', () => {
  const baseFrontmatter = `
agentId: configured-agent
fsm:
  states:
    - name: react
      type: llm`

  it('loads an optional non-negative integer max_tool_calls', () => {
    const config = loadAgentFromFrontmatter(`${baseFrontmatter}
  max_tool_calls: 2`)

    expect(config.fsm.max_tool_calls).toBe(2)
  })

  it('rejects invalid max_tool_calls values', () => {
    for (const value of ['-1', '1.5', 'unlimited']) {
      expect(() => loadAgentFromFrontmatter(`${baseFrontmatter}
  max_tool_calls: ${value}`)).toThrow('fsm.max_tool_calls')
    }
  })

  it('leaves max_tool_calls undefined when omitted', () => {
    expect(loadAgentFromFrontmatter(baseFrontmatter).fsm.max_tool_calls).toBeUndefined()
  })
})

describe('AgentRuntime tool call budgets', () => {
  it('shares a serial budget across tool batches and returns a stable error without executing the rejected handler', async () => {
    const executed: string[] = []
    const requests: ModelRequest[] = []
    const gateway: IModelGateway = {
      async complete(request: ModelRequest): Promise<ModelResponse> {
        requests.push(request)
        if (requests.length === 1) {
          return {
            content: [
              { type: 'tool_use', id: 'first', name: 'first', input: {} },
              { type: 'tool_use', id: 'second', name: 'second', input: {} },
            ],
            toolCalls: [
              { id: 'first', name: 'first', input: {} },
              { id: 'second', name: 'second', input: {} },
            ],
            finishReason: 'tool_use',
          }
        }
        if (requests.length === 2) return toolCallResponse('third', 'third', {})
        return textResponse('done')
      },
      async *stream(_request: ModelRequest): AsyncIterable<never> {
        yield* []
      },
    }
    const tool = (name: string): ToolDefinition => ({
      name,
      description: name,
      inputSchema: { type: 'object', properties: {} },
      handler: async () => {
        executed.push(name)
        return name
      },
    })
    const runtime = new AgentRuntime({
      config: makeConfig({ fsm: { states: [{ name: 'react', type: 'llm' }], max_tool_calls: 2 } }),
      goal: 'budget test',
      input: 'go',
      stateStore: new MemoryStore(),
      recorder: new InMemoryRecorder(),
      ioPort: new DefaultIOPort(gateway),
      extraTools: [tool('first'), tool('second'), tool('third')],
    })

    await runtime.run('go')

    expect(executed).toEqual(['first', 'second'])
    const rejected = requests[2]!.messages
      .flatMap(message => message.content)
      .find(content => content.type === 'tool_result' && content.tool_use_id === 'third') as { content: string; is_error: boolean }
    expect(rejected.is_error).toBe(true)
    expect(JSON.parse(rejected.content).code).toBe('TOOL_CALL_BUDGET_EXCEEDED')
  })

  it('reserves parallel slots before dispatching handlers', async () => {
    const parallelHandler = jest.fn(async () => 'ok')
    const requests: ModelRequest[] = []
    const gateway: IModelGateway = {
      async complete(request: ModelRequest): Promise<ModelResponse> {
        requests.push(request)
        if (requests.length === 1) {
          const toolCalls = ['one', 'two', 'three'].map(id => ({ id, name: 'parallel', input: { id } }))
          return {
            content: toolCalls.map(call => ({ type: 'tool_use' as const, ...call })),
            toolCalls,
            finishReason: 'tool_use',
          }
        }
        return textResponse('done')
      },
      async *stream(_request: ModelRequest): AsyncIterable<never> {
        yield* []
      },
    }
    const runtime = new AgentRuntime({
      config: makeConfig({ fsm: { states: [{ name: 'react', type: 'llm' }], max_tool_calls: 2 } }),
      goal: 'budget test',
      input: 'go',
      stateStore: new MemoryStore(),
      recorder: new InMemoryRecorder(),
      ioPort: new DefaultIOPort(gateway),
      extraTools: [{
        name: 'parallel',
        description: 'parallel',
        inputSchema: { type: 'object', properties: {} },
        parallelSafe: true,
        handler: parallelHandler,
      }],
    })

    await runtime.run('go')

    expect(parallelHandler).toHaveBeenCalledTimes(2)
    const rejected = requests[1]!.messages
      .flatMap(message => message.content)
      .find(content => content.type === 'tool_result' && content.tool_use_id === 'three') as { content: string; is_error: boolean }
    expect(rejected.is_error).toBe(true)
    expect(JSON.parse(rejected.content).code).toBe('TOOL_CALL_BUDGET_EXCEEDED')
  })

  it('rejects every handler when max_tool_calls is zero', async () => {
    const handler = jest.fn(async () => 'unexpected')
    const gateway = new SequentialGateway([
      toolCallResponse('zero', 'zero', {}),
      textResponse('done'),
    ])
    const runtime = new AgentRuntime({
      config: makeConfig({ fsm: { states: [{ name: 'react', type: 'llm' }], max_tool_calls: 0 } }),
      goal: 'budget test',
      input: 'go',
      stateStore: new MemoryStore(),
      recorder: new InMemoryRecorder(),
      ioPort: new DefaultIOPort(gateway),
      extraTools: [{
        name: 'zero',
        description: 'zero',
        inputSchema: { type: 'object', properties: {} },
        handler,
      }],
    })

    await runtime.run('go')

    expect(handler).not.toHaveBeenCalled()
  })

  it('records a budget rejection as a traceable tool error', async () => {
    const handler = jest.fn(async () => 'unexpected')
    const eventStore = new MemoryEventStore()
    const runtime = new AgentRuntime({
      config: makeConfig({ fsm: { states: [{ name: 'react', type: 'llm' }], max_tool_calls: 0 } }),
      goal: 'budget test',
      input: 'go',
      stateStore: new MemoryStore(),
      eventStore,
      recorder: new InMemoryRecorder('trace-budget', 'test-agent'),
      ioPort: new RecordingIOPort(
        new DefaultIOPort(new SequentialGateway([
          toolCallResponse('budgeted', 'budgeted', {}),
          textResponse('done'),
        ])),
        eventStore,
        'trace-budget',
      ),
      extraTools: [{
        name: 'budgeted',
        description: 'budgeted',
        inputSchema: { type: 'object', properties: {} },
        handler,
      }],
    })

    await runtime.run('go')

    expect(handler).not.toHaveBeenCalled()
    const response = (await eventStore.readByRunId('trace-budget'))
      .find(event => event.type === 'tool.responded')!.payload as {
        status: string
        error?: { code?: string }
      }
    expect(response.status).toBe('error')
    expect(response.error?.code).toBe('TOOL_CALL_BUDGET_EXCEEDED')
  })

  it('counts retryable handler attempts against the budget and records the rejected retry', async () => {
    const handler = jest.fn(async () => {
      throw Object.assign(new Error('retry me'), { retryable: true })
    })
    const eventStore = new MemoryEventStore()
    const runtime = new AgentRuntime({
      config: makeConfig({ fsm: { states: [{ name: 'react', type: 'llm' }], max_tool_calls: 1 } }),
      goal: 'budget test',
      input: 'go',
      stateStore: new MemoryStore(),
      eventStore,
      recorder: new InMemoryRecorder('trace-retry-budget', 'test-agent'),
      ioPort: new RecordingIOPort(
        new DefaultIOPort(new SequentialGateway([
          toolCallResponse('retryable', 'retryable', {}),
          textResponse('done'),
        ])),
        eventStore,
        'trace-retry-budget',
      ),
      extraTools: [{
        name: 'retryable',
        description: 'retryable',
        inputSchema: { type: 'object', properties: {} },
        handler,
      }],
    })

    await runtime.run('go')

    expect(handler).toHaveBeenCalledTimes(1)
    const responses = (await eventStore.readByRunId('trace-retry-budget'))
      .filter(event => event.type === 'tool.responded')
      .map(event => event.payload as { error?: { code?: string } })
    expect(responses).toHaveLength(2)
    expect(responses.at(-1)?.error?.code).toBe('TOOL_CALL_BUDGET_EXCEEDED')
  })

  it('blocks action-state handlers at budget zero and records the rejection', async () => {
    const handler = jest.fn(async () => 'unexpected')
    const eventStore = new MemoryEventStore()
    const runtime = new AgentRuntime({
      config: makeConfig({
        fsm: {
          states: [{ name: 'act', type: 'action', handler: 'action' }],
          max_tool_calls: 0,
        },
      }),
      goal: 'budget test',
      input: 'go',
      stateStore: new MemoryStore(),
      eventStore,
      recorder: new InMemoryRecorder('trace-action-budget', 'test-agent'),
      ioPort: new RecordingIOPort(
        new DefaultIOPort(new SequentialGateway([])),
        eventStore,
        'trace-action-budget',
      ),
      extraTools: [{
        name: 'action',
        description: 'action',
        inputSchema: { type: 'object', properties: {} },
        handler,
      }],
    })

    const result = await runtime.run('go')

    expect(handler).not.toHaveBeenCalled()
    expect(result.status).toBe('error')
    const response = (await eventStore.readByRunId('trace-action-budget'))
      .find(event => event.type === 'tool.responded')!.payload as {
        status: string
        error?: { code?: string }
      }
    expect(response.status).toBe('error')
    expect(response.error?.code).toBe('TOOL_CALL_BUDGET_EXCEEDED')
  })

  it('keeps tool dispatch unlimited when max_tool_calls is omitted', async () => {
    const handler = jest.fn(async () => 'ok')
    const gateway = new SequentialGateway([
      toolCallResponse('unlimited-one', 'unlimited', {}),
      toolCallResponse('unlimited-two', 'unlimited', {}),
      textResponse('done'),
    ])
    const runtime = new AgentRuntime({
      config: makeConfig(),
      goal: 'budget test',
      input: 'go',
      stateStore: new MemoryStore(),
      recorder: new InMemoryRecorder(),
      ioPort: new DefaultIOPort(gateway),
      extraTools: [{
        name: 'unlimited',
        description: 'unlimited',
        inputSchema: { type: 'object', properties: {} },
        handler,
      }],
    })

    await runtime.run('go')

    expect(handler).toHaveBeenCalledTimes(2)
  })
})
