import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { createServiceLogger } from '../logging/logger'
import type { AgentConfig } from '../types/agent'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'

function memorySink(): { lines: () => Record<string, unknown>[]; stream: { write: (s: string) => void } } {
  const raw: string[] = []
  return {
    lines: () => raw.flatMap(s => s.split('\n').filter(Boolean)).map(s => JSON.parse(s) as Record<string, unknown>),
    stream: { write: (s: string) => { raw.push(s) } },
  }
}

const AGENT: AgentConfig = {
  agentId: 'echo', version: '1.0.0', systemPrompt: 'echo',
  fsm: { states: [{ name: 'react', type: 'llm' }] },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
}

function okGateway(): IModelGateway {
  return {
    async complete(_req: ModelRequest): Promise<ModelResponse> {
      return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'end_turn' }
    },
    async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
      yield { type: 'message_delta', data: { text: 'done' } }
    },
  }
}

function failGateway(): IModelGateway {
  return {
    async complete(): Promise<ModelResponse> { throw new Error('provider down') },
    // eslint-disable-next-line require-yield
    async *stream(): AsyncIterable<ModelEvent> { throw new Error('provider down') },
  }
}

describe('Milkie.invoke service log', () => {
  it('logs one info summary per invoke: mod/agentId/runId/contextId/durationMs/status', async () => {
    const sink = memorySink()
    const milkie = new Milkie({
      stateStore: new MemoryStore(), gateway: okGateway(),
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    milkie.registerAgent(AGENT)
    const result = await milkie.invoke({ agentId: 'echo', goal: 'g', input: 'hi', contextId: 'ctx-1' })
    const summaries = sink.lines().filter(l => l.msg === 'invoke completed')
    expect(summaries).toHaveLength(1)
    const line = summaries[0]!
    expect(line.mod).toBe('runtime')
    expect(line.agentId).toBe('echo')
    expect(line.runId).toBe(result.agentRunId)
    expect(line.contextId).toBe('ctx-1')
    expect(line.status).toBe('completed')
    expect(typeof line.durationMs).toBe('number')
  })

  it('concurrent invokes keep their own runId/contextId（不串线）', async () => {
    const sink = memorySink()
    const milkie = new Milkie({
      stateStore: new MemoryStore(), gateway: okGateway(),
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    milkie.registerAgent(AGENT)
    const [a, b] = await Promise.all([
      milkie.invoke({ agentId: 'echo', goal: 'g', input: 'x', contextId: 'ctx-a' }),
      milkie.invoke({ agentId: 'echo', goal: 'g', input: 'y', contextId: 'ctx-b' }),
    ])
    const byCtx = Object.fromEntries(
      sink.lines().filter(l => l.msg === 'invoke completed').map(l => [l.contextId, l.runId]),
    )
    expect(byCtx['ctx-a']).toBe(a.agentRunId)
    expect(byCtx['ctx-b']).toBe(b.agentRunId)
  })

  it('LLM 持续失败时仍出一条 invoke 汇总（AgentRuntime 把错误吞成 status:error）', async () => {
    const sink = memorySink()
    const milkie = new Milkie({
      stateStore: new MemoryStore(), gateway: failGateway(),
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    milkie.registerAgent(AGENT)
    const result = await milkie.invoke({ agentId: 'echo', goal: 'g', input: 'hi', contextId: 'ctx-err' })
    expect(result.status).toBe('error')
    const summaries = sink.lines().filter(l => typeof l.msg === 'string' && (l.msg as string).startsWith('invoke'))
    expect(summaries).toHaveLength(1)
    expect(summaries[0]!.status).toBe('error')
  })

  it('tier 回退从 console.debug 收编为 logger.warn', async () => {
    const sink = memorySink()
    const milkie = new Milkie({
      stateStore: new MemoryStore(), gateway: okGateway(),
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    milkie.registerAgent(AGENT)
    await milkie.complete('echo', { messages: [{ role: 'user', content: [{ type: 'text', text: 'q' }] }], tier: 'no-such-tier' })
    const warns = sink.lines().filter(l => l.level === 'warn')
    expect(warns.length).toBeGreaterThanOrEqual(1)
    expect(warns[0]!.tier).toBe('no-such-tier')
    expect(warns[0]!.agentId).toBe('echo')
  })
})
