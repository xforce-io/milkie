import { AgentRuntime } from '../runtime/AgentRuntime'
import { DefaultIOPort } from '../runtime/IOPort'
import { MemoryStore } from '../store/MemoryStore'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'
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

/**
 * A gateway whose `stream` yields a fixed sequence of ModelEvent batches (one
 * batch per LLM round), and whose `complete` returns the text-equivalent
 * ModelResponse for the same round. This lets the same gateway be driven either
 * streaming (onEvent provided → DefaultIOPort calls stream) or non-streaming
 * (onEvent absent → DefaultIOPort calls complete), and asserts the runtime's
 * behavior is equivalent across both paths.
 */
class ScriptedStreamGateway implements IModelGateway {
  private streamIndex = 0
  private completeIndex = 0

  constructor(
    private readonly rounds: ModelEvent[][],
    private readonly completions: ModelResponse[],
  ) {}

  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.completions[this.completeIndex++]
    if (!r) throw new Error('No more mock completions')
    return r
  }

  async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
    const batch = this.rounds[this.streamIndex++]
    if (!batch) throw new Error('No more mock stream rounds')
    for (const ev of batch) {
      yield ev
    }
  }
}

function textCompletion(text: string): ModelResponse {
  return {
    content:      [{ type: 'text', text }],
    toolCalls:    [],
    finishReason: 'end_turn',
  }
}

// ---- Tests ----

describe('AgentRuntime — onModelEvent stream pass-through (#80)', () => {
  it('forwards message_delta events to onModelEvent and completes', async () => {
    // One round: two token deltas that aggregate to "Hello, world!"
    const gateway = new ScriptedStreamGateway(
      [[
        { type: 'message_delta', data: { text: 'Hello, ' } },
        { type: 'message_delta', data: { text: 'world!' } },
      ]],
      [textCompletion('Hello, world!')],
    )

    const events: ModelEvent[] = []
    const runtime = new AgentRuntime({
      config:       makeConfig(),
      goal:         'test goal',
      input:        'hi',
      stateStore:   new MemoryStore(),
      recorder:     new InMemoryRecorder(undefined, 'test-agent'),
      ioPort:       new DefaultIOPort(gateway),
      onModelEvent: (e) => events.push(e),
    })

    const result = await runtime.run('hi')

    expect(result.status).toBe('completed')
    expect(result.output).toBe('Hello, world!')

    const deltas = events.filter(e => e.type === 'message_delta')
    expect(deltas).toHaveLength(2)
    expect(deltas.map(d => (d as Extract<ModelEvent, { type: 'message_delta' }>).data.text))
      .toEqual(['Hello, ', 'world!'])
  })

  it('behaves identically (completed, same output) when onModelEvent is omitted', async () => {
    // No onModelEvent → DefaultIOPort uses complete(); stream() is never touched.
    const gateway = new ScriptedStreamGateway(
      [[
        { type: 'message_delta', data: { text: 'Hello, ' } },
        { type: 'message_delta', data: { text: 'world!' } },
      ]],
      [textCompletion('Hello, world!')],
    )

    const runtime = new AgentRuntime({
      config:     makeConfig(),
      goal:       'test goal',
      input:      'hi',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(undefined, 'test-agent'),
      ioPort:     new DefaultIOPort(gateway),
      // onModelEvent intentionally omitted
    })

    const result = await runtime.run('hi')

    expect(result.status).toBe('completed')
    expect(result.output).toBe('Hello, world!')
  })

  it('forwards tool_call_* stream events through the runtime across a tool round', async () => {
    const searchTool: ToolDefinition = {
      name:        'search',
      description: 'search the web',
      inputSchema: { type: 'object', properties: { q: { type: 'string' } } },
      parallelSafe: true,
      handler:     async () => ({ results: ['result1'] }),
    }

    // Round 1: a streamed tool call (start → delta → done).
    // Round 2: a streamed text answer.
    const gateway = new ScriptedStreamGateway(
      [
        [
          { type: 'tool_call_start', data: { toolCallId: 'tc-1', name: 'search' } },
          { type: 'tool_call_delta', data: { toolCallId: 'tc-1', delta: '{"q":"test"}' } },
          { type: 'tool_call_done', data: { toolCallId: 'tc-1', input: { q: 'test' } } },
        ],
        [
          { type: 'message_delta', data: { text: 'I found ' } },
          { type: 'message_delta', data: { text: 'result1' } },
        ],
      ],
      [
        // complete-path equivalents (unused here, but keep gateway symmetric)
        {
          content:   [{ type: 'tool_use', id: 'tc-1', name: 'search', input: { q: 'test' } }],
          toolCalls: [{ id: 'tc-1', name: 'search', input: { q: 'test' } }],
          finishReason: 'tool_use',
        },
        textCompletion('I found result1'),
      ],
    )

    const events: ModelEvent[] = []
    const runtime = new AgentRuntime({
      config:       makeConfig(),
      goal:         'search something',
      input:        'search for test',
      stateStore:   new MemoryStore(),
      recorder:     new InMemoryRecorder(),
      ioPort:       new DefaultIOPort(gateway),
      extraTools:   [searchTool],
      onModelEvent: (e) => events.push(e),
    })

    const result = await runtime.run('search for test')

    expect(result.status).toBe('completed')
    expect(result.output).toBe('I found result1')

    // The tool-call stream from round 1 must have flowed through the runtime.
    expect(events.some(e => e.type === 'tool_call_start')).toBe(true)
    expect(events.some(e => e.type === 'tool_call_delta')).toBe(true)
    expect(events.some(e => e.type === 'tool_call_done')).toBe(true)
    // And round 2's text deltas too.
    expect(events.filter(e => e.type === 'message_delta')).toHaveLength(2)
  })
})
