/**
 * #164: ToolContext.currentTurn — runtime exposes the current turn input to tool handlers.
 */
import { AgentRuntime } from '../runtime/AgentRuntime'
import { DefaultIOPort } from '../runtime/IOPort'
import { MemoryStore } from '../store/MemoryStore'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { ToolDefinition } from '../types/tool'

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

function textResponse(text: string): ModelResponse {
  return { content: [{ type: 'text', text }], toolCalls: [], finishReason: 'end_turn' }
}

function toolCallResponse(id: string, name: string, input: unknown): ModelResponse {
  return {
    content:      [{ type: 'tool_use', id, name, input }],
    toolCalls:    [{ id, name, input }],
    finishReason: 'tool_use',
  }
}

// ---- Tests ----

describe('#164 ToolContext.currentTurn', () => {
  it('LLM-state tool handler receives ctx.currentTurn matching the turn input', async () => {
    let capturedCurrentTurn: string | undefined

    const tool: ToolDefinition = {
      name:        'probe',
      description: 'capture currentTurn',
      inputSchema: { type: 'object', properties: {} },
      handler:     async (_input, ctx) => {
        capturedCurrentTurn = ctx.currentTurn
        return { ok: true }
      },
    }

    const gateway = new SequentialGateway([
      toolCallResponse('tc-1', 'probe', {}),
      textResponse('done'),
    ])

    const runtime = new AgentRuntime({
      config:     makeConfig(),
      goal:       'test goal',
      input:      'hello world',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(gateway),
      extraTools: [tool],
    })

    await runtime.run('hello world')

    // currentTurn is built from: `Goal: ${goal}\n\n${input}`
    expect(capturedCurrentTurn).toBeDefined()
    expect(capturedCurrentTurn).toContain('hello world')
    expect(capturedCurrentTurn).toContain('test goal')
  })

  it('action-state handler receives ctx.currentTurn matching the turn input', async () => {
    let capturedCurrentTurn: string | undefined

    const actionTool: ToolDefinition = {
      name:        'action_probe',
      description: 'action handler capturing currentTurn',
      inputSchema: { type: 'object', properties: {} },
      handler:     async (_input, ctx) => {
        capturedCurrentTurn = ctx.currentTurn
        ctx.emit('DONE')
        return {}
      },
    }

    const config = makeConfig({
      fsm: {
        states: [
          {
            name:  'classify',
            type:  'llm',
            tools: ['trigger_action'],
            on:    { TRIGGER: 'process' },
          },
          {
            name:     'process',
            type:     'action',
            handler:  'action_probe',
            terminal: true,
          },
        ],
      },
    })

    const triggerTool: ToolDefinition = {
      name:        'trigger_action',
      description: 'trigger the action state',
      inputSchema: { type: 'object', properties: {} },
      handler:     async (_input, ctx) => {
        ctx.emit('TRIGGER')
        return {}
      },
    }

    const gateway = new SequentialGateway([
      toolCallResponse('tc-1', 'trigger_action', {}),
    ])

    const runtime = new AgentRuntime({
      config,
      goal:       'action goal',
      input:      'trigger the action',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(gateway),
      extraTools: [triggerTool, actionTool],
    })

    await runtime.run('trigger the action')

    expect(capturedCurrentTurn).toBeDefined()
    expect(capturedCurrentTurn).toContain('trigger the action')
    expect(capturedCurrentTurn).toContain('action goal')
  })

  it('ctx.currentTurn is identical across multiple tool-loop iterations in the same turn', async () => {
    const capturedTurns: Array<string | undefined> = []

    const tool: ToolDefinition = {
      name:        'multi_probe',
      description: 'capture currentTurn across calls',
      inputSchema: { type: 'object', properties: {} },
      handler:     async (_input, ctx) => {
        capturedTurns.push(ctx.currentTurn)
        return { call: capturedTurns.length }
      },
    }

    const gateway = new SequentialGateway([
      toolCallResponse('tc-1', 'multi_probe', {}),
      toolCallResponse('tc-2', 'multi_probe', {}),
      toolCallResponse('tc-3', 'multi_probe', {}),
      textResponse('all done'),
    ])

    const runtime = new AgentRuntime({
      config:     makeConfig(),
      goal:       'stable goal',
      input:      'run three times',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(gateway),
      extraTools: [tool],
    })

    await runtime.run('run three times')

    expect(capturedTurns).toHaveLength(3)
    expect(capturedTurns[0]).toBeDefined()
    // All three calls should receive the identical currentTurn value
    expect(capturedTurns[1]).toBe(capturedTurns[0])
    expect(capturedTurns[2]).toBe(capturedTurns[0])
    expect(capturedTurns[0]).toContain('run three times')
  })
})
