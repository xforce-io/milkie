import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'

class CapturingGateway implements IModelGateway {
  public requests: ModelRequest[] = []
  private index = 0

  constructor(private readonly responses: string[]) {}

  async complete(req: ModelRequest): Promise<ModelResponse> {
    this.requests.push(req)
    const text = this.responses[this.index++] ?? 'done'
    return {
      content:      [{ type: 'text', text }],
      toolCalls:    [],
      finishReason: 'end_turn',
    }
  }

  async *stream(_req: ModelRequest): AsyncIterable<never> {
    yield* []
  }
}

function makeConfig(): AgentConfig {
  return {
    agentId:      'projection-agent',
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
  }
}

function textOf(req: ModelRequest): string[] {
  return req.messages.map((m) => {
    const first = m.content[0]
    return first?.type === 'text' ? first.text : ''
  })
}

describe('Milkie context projections (#146)', () => {
  it('attaches projections per target context, dedupes by sourceRunId, and keeps the newest maxCount', async () => {
    const milkie = new Milkie({ stateStore: new MemoryStore(), gateway: new CapturingGateway([]) })

    await milkie.attachProjection('channel-c1', {
      sourceRunId:     'run-1',
      sourceContextId: 'job-c',
      displayText:     'old run 1',
      deliveredAt:     1000,
    })
    await milkie.attachProjection('channel-c1', {
      sourceRunId: 'run-2',
      displayText: 'run 2',
      deliveredAt: 2000,
    })
    await milkie.attachProjection('channel-c1', {
      sourceRunId: 'run-1',
      displayText: 'updated run 1',
      deliveredAt: 3000,
    })
    await milkie.attachProjection('channel-c1', {
      sourceRunId: 'run-3',
      displayText: 'run 3',
      deliveredAt: 4000,
      bound:       { maxCount: 2 },
    })
    await milkie.attachProjection('other-context', {
      sourceRunId: 'run-x',
      displayText: 'other',
      deliveredAt: 5000,
    })

    expect(await milkie.listContextProjections('channel-c1')).toEqual([
      expect.objectContaining({ sourceRunId: 'run-1', displayText: 'updated run 1', deliveredAt: 3000 }),
      expect.objectContaining({ sourceRunId: 'run-3', displayText: 'run 3', deliveredAt: 4000 }),
    ])
    expect(await milkie.listContextProjections('other-context')).toEqual([
      expect.objectContaining({ sourceRunId: 'run-x', displayText: 'other' }),
    ])
  })

  it('rejects a non-positive maxCount bound', async () => {
    const milkie = new Milkie({ stateStore: new MemoryStore(), gateway: new CapturingGateway([]) })

    await expect(milkie.attachProjection('channel-c1', {
      sourceRunId: 'run-1',
      displayText: 'report',
      bound:       { maxCount: 0 },
    })).rejects.toThrow(/maxCount/)
  })

  it('merges delivered projections into the current-turn user message (input first), never as a standalone turn between history and the reply (#192)', async () => {
    const gateway = new CapturingGateway(['first answer', 'second answer'])
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway,
    })
    milkie.registerAgent(makeConfig())

    await milkie.invoke({
      agentId:   'projection-agent',
      goal:      'remember first turn',
      input:     'first question',
      contextId: 'channel-c1',
    })

    await milkie.attachProjection('channel-c1', {
      sourceRunId:     'job-run-1',
      sourceContextId: 'job-context',
      displayText:     'Nightly report: all systems green.',
      summary:         'systems green',
      deliveredAt:     12345,
    })

    await milkie.invoke({
      agentId:   'projection-agent',
      goal:      'answer followup',
      input:     'what did the report say?',
      contextId: 'channel-c1',
    })

    const secondRequest = gateway.requests[1]!
    const messages = textOf(secondRequest)

    // history is preserved
    expect(messages[0]).toContain('first question')
    expect(messages[1]).toContain('first answer')

    // C1: the current user input is adjacent to the last history assistant —
    // no standalone projection turn wedged between them.
    expect(messages).toHaveLength(3)
    expect(secondRequest.messages[2]!.role).toBe('user')

    // C2/C3: projection is merged into the current-turn message; the user's real
    // input comes first, the delivered-context block follows, clearly labeled.
    const cur = messages[2]!
    expect(cur).toContain('what did the report say?')
    expect(cur).toContain('External Delivered Context')
    expect(cur).toContain('not a user message')
    expect(cur).toContain('Nightly report: all systems green.')
    expect(cur).toContain('job-run-1')
    expect(cur).toContain('job-context')
    expect(cur.indexOf('what did the report say?'))
      .toBeLessThan(cur.indexOf('Nightly report: all systems green.'))

    // projection never appears as its own message — only the current-turn carries it
    const carrying = messages.filter((t) => t.includes('Nightly report: all systems green.'))
    expect(carrying).toHaveLength(1)

    // system block stays free of projection content
    expect(secondRequest.system).not.toContain('Nightly report')
  })

  it('does not materialize projections into the target context session history', async () => {
    const gateway = new CapturingGateway(['first answer', 'second answer'])
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway,
    })
    milkie.registerAgent(makeConfig())

    await milkie.invoke({
      agentId:   'projection-agent',
      goal:      'first',
      input:     'first question',
      contextId: 'channel-c1',
    })
    await milkie.attachProjection('channel-c1', {
      sourceRunId: 'job-run-1',
      displayText: 'Report that must not become chat history.',
      deliveredAt: 12345,
    })
    await milkie.invoke({
      agentId:   'projection-agent',
      goal:      'followup',
      input:     'second question',
      contextId: 'channel-c1',
    })

    const transcriptText = (await milkie.getSessionHistory('channel-c1'))
      .flatMap(m => m.content)
      .filter((c): c is { type: 'text'; text: string } => c.type === 'text')
      .map(c => c.text)
      .join('\n')

    expect(transcriptText).toContain('first question')
    expect(transcriptText).toContain('second question')
    expect(transcriptText).not.toContain('Report that must not become chat history.')
  })
})
