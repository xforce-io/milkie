import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'
import type { ToolDefinition } from '../types/tool'

// ---- Fixtures ----

function makeConfig(overrides: Partial<AgentConfig> = {}): AgentConfig {
  return {
    agentId:      'stream-agent',
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
 * ModelResponse for the same round. Drives either path depending on whether
 * the IOPort receives an onModelEvent callback (stream) or not (complete).
 * Records which path was taken so tests can assert the non-streaming branch
 * never touches stream().
 */
class ScriptedStreamGateway implements IModelGateway {
  private streamIndex = 0
  private completeIndex = 0
  public completeCalls = 0
  public streamCalls = 0

  constructor(
    private readonly rounds: ModelEvent[][],
    private readonly completions: ModelResponse[],
  ) {}

  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.completeCalls++
    const r = this.completions[this.completeIndex++]
    if (!r) throw new Error('No more mock completions')
    return r
  }

  async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
    this.streamCalls++
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

describe('Milkie.invoke — onModelEvent end-to-end pass-through (#80)', () => {
  it('streams message_delta events to onModelEvent and returns the aggregated output', async () => {
    const gateway = new ScriptedStreamGateway(
      [[
        { type: 'message_delta', data: { text: 'Hello, ' } },
        { type: 'message_delta', data: { text: 'world!' } },
      ]],
      [textCompletion('Hello, world!')],
    )

    const milkie = new Milkie({ stateStore: new MemoryStore(), gateway })
    milkie.registerAgent(makeConfig())

    const events: ModelEvent[] = []
    const result = await milkie.invoke({
      agentId:      'stream-agent',
      goal:         'greet',
      input:        'hi',
      onModelEvent: (e) => events.push(e),
    })

    expect(result.status).toBe('completed')
    expect(result.output).toBe('Hello, world!')

    // The stream path was taken end-to-end; complete() was never used.
    expect(gateway.streamCalls).toBe(1)
    expect(gateway.completeCalls).toBe(0)

    const deltas = events.filter(e => e.type === 'message_delta')
    expect(deltas).toHaveLength(2)
    expect(deltas.map(d => (d as Extract<ModelEvent, { type: 'message_delta' }>).data.text))
      .toEqual(['Hello, ', 'world!'])
  })

  it('omitting onModelEvent uses the non-streaming complete() path with the same output', async () => {
    const gateway = new ScriptedStreamGateway(
      [[
        { type: 'message_delta', data: { text: 'Hello, ' } },
        { type: 'message_delta', data: { text: 'world!' } },
      ]],
      [textCompletion('Hello, world!')],
    )

    const milkie = new Milkie({ stateStore: new MemoryStore(), gateway })
    milkie.registerAgent(makeConfig())

    const result = await milkie.invoke({
      agentId: 'stream-agent',
      goal:    'greet',
      input:   'hi',
      // onModelEvent intentionally omitted
    })

    expect(result.status).toBe('completed')
    expect(result.output).toBe('Hello, world!')

    // No onModelEvent → DefaultIOPort calls complete(); stream() is never touched.
    expect(gateway.completeCalls).toBe(1)
    expect(gateway.streamCalls).toBe(0)
  })

  it('forwards tool_call_* stream events end-to-end across a tool round', async () => {
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
        {
          content:   [{ type: 'tool_use', id: 'tc-1', name: 'search', input: { q: 'test' } }],
          toolCalls: [{ id: 'tc-1', name: 'search', input: { q: 'test' } }],
          finishReason: 'tool_use',
        },
        textCompletion('I found result1'),
      ],
    )

    const milkie = new Milkie({ stateStore: new MemoryStore(), gateway, tools: [searchTool] })
    milkie.registerAgent(makeConfig())

    const events: ModelEvent[] = []
    const result = await milkie.invoke({
      agentId:      'stream-agent',
      goal:         'search something',
      input:        'search for test',
      onModelEvent: (e) => events.push(e),
    })

    expect(result.status).toBe('completed')
    expect(result.output).toBe('I found result1')

    // The tool-call stream from round 1 must have flowed through end-to-end.
    expect(events.some(e => e.type === 'tool_call_start')).toBe(true)
    expect(events.some(e => e.type === 'tool_call_delta')).toBe(true)
    expect(events.some(e => e.type === 'tool_call_done')).toBe(true)
    // And round 2's text deltas too.
    expect(events.filter(e => e.type === 'message_delta')).toHaveLength(2)
    // Streamed both rounds, never fell back to complete().
    expect(gateway.streamCalls).toBe(2)
    expect(gateway.completeCalls).toBe(0)
  })
})
