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
})
