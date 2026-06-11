import { LoggingGateway } from '../logging/LoggingGateway'
import { createServiceLogger } from '../logging/logger'
import { createGateway } from '../gateway/GatewayFactory'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'

function memorySink(): { lines: () => Record<string, unknown>[]; stream: { write: (s: string) => void } } {
  const raw: string[] = []
  return {
    lines: () => raw.flatMap(s => s.split('\n').filter(Boolean)).map(s => JSON.parse(s) as Record<string, unknown>),
    stream: { write: (s: string) => { raw.push(s) } },
  }
}

const REQ: ModelRequest = { model: 'test-model', messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] }

function okGateway(): IModelGateway {
  return {
    async complete(): Promise<ModelResponse> {
      return {
        content: [{ type: 'text', text: 'ok' }], toolCalls: [],
        usage: { inputTokens: 11, outputTokens: 7 }, finishReason: 'end_turn',
      }
    },
    async *stream(): AsyncIterable<ModelEvent> {
      yield { type: 'message_delta', data: { text: 'ok' } }
      yield { type: 'usage', data: { inputTokens: 3, outputTokens: 5 } }
    },
  }
}

function failGateway(): IModelGateway {
  return {
    async complete(): Promise<ModelResponse> { throw new Error('rate limited') },
    // eslint-disable-next-line require-yield
    async *stream(): AsyncIterable<ModelEvent> { throw new Error('socket reset') },
  }
}

describe('LoggingGateway.complete', () => {
  it('logs one info wide event with model/durationMs/tokens on success', async () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    const gw = new LoggingGateway(okGateway(), log)
    await gw.complete(REQ)
    const line = sink.lines()[0]!
    expect(line.msg).toBe('llm call')
    expect(line.level).toBe('info')
    expect(line.model).toBe('test-model')
    expect(typeof line.durationMs).toBe('number')
    expect(line.inputTokens).toBe(11)
    expect(line.outputTokens).toBe(7)
  })

  it('logs error with err.stack and rethrows on failure', async () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    const gw = new LoggingGateway(failGateway(), log)
    await expect(gw.complete(REQ)).rejects.toThrow('rate limited')
    const line = sink.lines()[0]!
    expect(line.level).toBe('error')
    expect(line.msg).toBe('llm call failed')
    expect((line.err as { message: string }).message).toBe('rate limited')
  })

  it('does not log prompt or completion content (脱敏)', async () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    await new LoggingGateway(okGateway(), log).complete(REQ)
    const serialized = JSON.stringify(sink.lines())
    expect(serialized).not.toContain('"hi"')   // prompt 正文
    expect(serialized).not.toContain('"ok"')   // completion 正文
  })
})

describe('LoggingGateway.stream', () => {
  it('passes events through and logs one summary with accumulated usage', async () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    const gw = new LoggingGateway(okGateway(), log)
    const events: ModelEvent[] = []
    for await (const e of gw.stream(REQ)) events.push(e)
    expect(events).toHaveLength(2)                       // 事件原样透传
    const line = sink.lines()[0]!
    expect(line.msg).toBe('llm stream')
    expect(line.model).toBe('test-model')
    expect(line.inputTokens).toBe(3)
    expect(line.outputTokens).toBe(5)
  })

  it('logs error and rethrows when the stream throws', async () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    const gw = new LoggingGateway(failGateway(), log)
    await expect(async () => { for await (const _ of gw.stream(REQ)) { /* drain */ } }).rejects.toThrow('socket reset')
    const line = sink.lines()[0]!
    expect(line.level).toBe('error')
    expect(line.msg).toBe('llm stream failed')
  })
})

describe('createGateway wiring', () => {
  it('returns a LoggingGateway-wrapped adapter', () => {
    const gw = createGateway({ provider: 'anthropic', model: 'claude-x', adapter: 'anthropic' })
    expect(gw).toBeInstanceOf(LoggingGateway)
  })
})
