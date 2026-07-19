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
    agentId:      'truncate-tester',
    version:      '1.0.0',
    systemPrompt: 'test',
    fsm: {
      states: [{ name: 'react', type: 'llm', max_iterations: 5 }],
    },
    model: {
      provider: 'stub',
      model:    'stub',
      adapter:  'test',
    },
    ...overrides,
  }
}

/**
 * A gateway that captures every request it sees, returns a tool_call on the
 * first invocation, and a text response on the second.
 */
function makeRequestCapturingGateway(toolName: string): {
  gateway: IModelGateway
  requestsSeen: ModelRequest[]
} {
  const requestsSeen: ModelRequest[] = []

  const gateway: IModelGateway = {
    async complete(req: ModelRequest): Promise<ModelResponse> {
      requestsSeen.push(req)
      if (requestsSeen.length === 1) {
        return {
          content:      [{ type: 'tool_use', id: 'tc1', name: toolName, input: {} }],
          toolCalls:    [{ id: 'tc1', name: toolName, input: {} }],
          finishReason: 'tool_use',
        }
      }
      return {
        content:      [{ type: 'text', text: 'done' }],
        toolCalls:    [],
        finishReason: 'end_turn',
      }
    },
    async *stream(_req: ModelRequest): AsyncIterable<never> {
      yield* []
    },
  }

  return { gateway, requestsSeen }
}

// ---- Tests ----

describe('AgentRuntime — ToolResultStrategy applied end-to-end', () => {
  test('tool with truncate(50) → tool_result content in next LLM request is truncated', async () => {
    const bigReadTool: ToolDefinition = {
      name:        'big_read',
      description: 'returns a large string',
      inputSchema: { type: 'object', properties: {}, required: [] },
      parallelSafe: true,
      handler: async () => 'X'.repeat(5000),
      resultStrategy: { shape: { kind: 'truncate', maxChars: 50 } },
    }

    const { gateway, requestsSeen } = makeRequestCapturingGateway('big_read')

    const runtime = new AgentRuntime({
      config:     makeConfig(),
      goal:       'test',
      input:      'go',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(gateway),
      extraTools: [bigReadTool],
    })

    const result = await runtime.run('go')
    expect(result.status).toBe('completed')

    // Two LLM calls: iteration 1 (tool call) + iteration 2 (text response)
    expect(requestsSeen.length).toBe(2)

    const messagesIter2 = requestsSeen[1]!.messages
    const toolResultMsg = messagesIter2.find(
      m => m.role === 'tool' && m.content.some(c => c.type === 'tool_result'),
    )
    expect(toolResultMsg).toBeDefined()

    const tr = toolResultMsg!.content.find(c => c.type === 'tool_result') as {
      type: 'tool_result'
      content: string
    }
    expect(tr).toBeDefined()

    // Truncated content must be short — not the raw 5000 chars
    expect(tr.content.length).toBeLessThan(100)

    // Starts with the repeated character and ends with the truncation indicator
    expect(tr.content.startsWith('XXXX')).toBe(true)
    expect(tr.content.endsWith('...')).toBe(true)
  })

  test('tool without resultStrategy → tool_result content unchanged (verbatim default)', async () => {
    const verbatimTool: ToolDefinition = {
      name:        'big_read_verbatim',
      description: 'returns a large string',
      inputSchema: { type: 'object', properties: {}, required: [] },
      parallelSafe: true,
      handler: async () => 'Y'.repeat(5000),
      // no resultStrategy — defaults to verbatim
    }

    const { gateway, requestsSeen } = makeRequestCapturingGateway('big_read_verbatim')

    const runtime = new AgentRuntime({
      config:     makeConfig(),
      goal:       'test',
      input:      'go',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(gateway),
      extraTools: [verbatimTool],
    })

    const result = await runtime.run('go')
    expect(result.status).toBe('completed')

    expect(requestsSeen.length).toBe(2)

    const messagesIter2 = requestsSeen[1]!.messages
    const toolResultMsg = messagesIter2.find(
      m => m.role === 'tool' && m.content.some(c => c.type === 'tool_result'),
    )
    expect(toolResultMsg).toBeDefined()

    const tr = toolResultMsg!.content.find(c => c.type === 'tool_result') as {
      type: 'tool_result'
      content: string
    }
    expect(tr).toBeDefined()

    // Verbatim: full 5000 chars must arrive unchanged
    expect(tr.content.length).toBe(5000)
  })

  test('built-in run_command default strategy shapes oversized stdout into LLM tool_result (alfred#160)', async () => {
    // Drive the *shipped* run_command definition (not a test double) so registration
    // and resultStrategy on execTools are what AgentRuntime applies.
    const { execTools, RUN_COMMAND_LLM_MAX_CHARS, runCommand } = await import('../tools/exec')
    const runCmdDef = execTools.find(t => t.name === 'run_command')!
    expect(runCmdDef.resultStrategy).toBeDefined()
    const shape = runCmdDef.resultStrategy!.shape
    expect(typeof shape === 'object' && shape.kind === 'tail').toBe(true)

    // Oversized payload: larger than LLM maxChars but under stream cap (~30k)
    const bigLen = RUN_COMMAND_LLM_MAX_CHARS + 5_000
    expect(bigLen).toBeLessThan(30_000)

    // Gateway: first call asks for run_command with a command that prints bigLen A's
    const requestsSeen: ModelRequest[] = []
    const gateway: IModelGateway = {
      async complete(req: ModelRequest): Promise<ModelResponse> {
        requestsSeen.push(req)
        if (requestsSeen.length === 1) {
          const cmd = `${process.execPath} -e "process.stdout.write('A'.repeat(${bigLen}))"`
          return {
            content:      [{ type: 'tool_use', id: 'tc-rc', name: 'run_command', input: { command: cmd } }],
            toolCalls:    [{ id: 'tc-rc', name: 'run_command', input: { command: cmd } }],
            finishReason: 'tool_use',
          }
        }
        return {
          content:      [{ type: 'text', text: 'done' }],
          toolCalls:    [],
          finishReason: 'end_turn',
        }
      },
      async *stream(_req: ModelRequest): AsyncIterable<never> {
        yield* []
      },
    }

    const recorder = new InMemoryRecorder()
    const runtime = new AgentRuntime({
      config:     makeConfig(),
      goal:       'test',
      input:      'go',
      stateStore: new MemoryStore(),
      recorder,
      ioPort:     new DefaultIOPort(gateway),
      // Register the real built-in tool only (avoid double-register of system tools if any)
      extraTools: [runCmdDef],
    })

    // Sanity: real handler produces raw stdout at least bigLen
    const rawOut = await runCommand({
      command: `${process.execPath} -e "process.stdout.write('A'.repeat(${bigLen}))"`,
    })
    expect(rawOut.stdout.length).toBe(bigLen)
    expect(rawOut.truncated).toBe(false)

    const result = await runtime.run('go')
    expect(result.status).toBe('completed')
    expect(requestsSeen.length).toBe(2)

    const messagesIter2 = requestsSeen[1]!.messages
    const toolResultMsg = messagesIter2.find(
      m => m.role === 'tool' && m.content.some(c => c.type === 'tool_result'),
    )
    expect(toolResultMsg).toBeDefined()
    const tr = toolResultMsg!.content.find(c => c.type === 'tool_result') as {
      type: 'tool_result'
      content: string
    }
    expect(tr).toBeDefined()

    // LLM-facing content must be shaped ≤ maxChars (+ small marker overhead for tail prefix)
    expect(tr.content.length).toBeLessThanOrEqual(RUN_COMMAND_LLM_MAX_CHARS + 80)
    expect(tr.content.length).toBeLessThan(bigLen)
    expect(tr.content).toMatch(/chars dropped/i)
    // tail keeps the end of the payload
    expect(tr.content.endsWith('AAAAA') || tr.content.includes('AAAAA')).toBe(true)

    // tool.shaped observation when bytes actually changed
    const shapedEvents = recorder.getSpans()
      .flatMap(s => s.events)
      .filter(e => e.name === 'tool.shaped')
    expect(shapedEvents.length).toBeGreaterThanOrEqual(1)
    const attrs = shapedEvents[0]!.attributes as {
      rawBytes: number
      storedBytes: number
      shapeKind: string
      toolName: string
    }
    expect(attrs.toolName).toBe('run_command')
    expect(attrs.shapeKind).toBe('tail')
    expect(attrs.rawBytes).toBeGreaterThan(attrs.storedBytes)
    expect(attrs.storedBytes).toBeLessThanOrEqual(RUN_COMMAND_LLM_MAX_CHARS + 80)
  })
})
