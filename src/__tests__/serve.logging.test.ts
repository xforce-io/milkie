import http from 'http'
import type { AddressInfo } from 'net'
import { createServeServer } from '../cli/serve'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { BroadcastingEventStore } from '../trace/BroadcastingEventStore'
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

function buildMilkie(): { milkie: Milkie; agentId: string; broadcaster: BroadcastingEventStore } {
  const broadcaster = new BroadcastingEventStore(new MemoryEventStore())
  const gateway: IModelGateway = {
    async complete(_req: ModelRequest): Promise<ModelResponse> {
      return { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'end_turn' }
    },
    async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
      yield { type: 'message_delta', data: { text: 'ok' } }
    },
  }
  const agent: AgentConfig = {
    agentId: 'echo', version: '1.0.0', systemPrompt: 'echo',
    fsm: { states: [{ name: 'react', type: 'llm' }] },
    model: { provider: 'stub', model: 'stub', adapter: 'stub' },
  }
  const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: broadcaster, gateway })
  milkie.registerAgent(agent)
  return { milkie, agentId: 'echo', broadcaster }
}

function listen(server: http.Server): Promise<number> {
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve((server.address() as AddressInfo).port)))
}

function get(port: number, path: string): Promise<number> {
  return new Promise((resolve, reject) => {
    http.get({ host: '127.0.0.1', port, path }, res => { res.resume(); res.on('end', () => resolve(res.statusCode ?? 0)) }).on('error', reject)
  })
}

function post(port: number, path: string, body: unknown): Promise<{ status: number; text: string }> {
  return new Promise((resolve, reject) => {
    const req = http.request(
      { host: '127.0.0.1', port, path, method: 'POST', headers: { 'content-type': 'application/json' } },
      res => {
        let text = ''
        res.on('data', c => { text += c })
        res.on('end', () => resolve({ status: res.statusCode ?? 0, text }))
      },
    )
    req.on('error', reject)
    req.end(JSON.stringify(body))
  })
}

/** res.on('finish') 的日志写入在响应抵达客户端后才跑，轮询等它出现。 */
async function waitFor<T>(read: () => T | undefined, ms = 2000): Promise<T> {
  const deadline = Date.now() + ms
  for (;;) {
    const v = read()
    if (v !== undefined) return v
    if (Date.now() > deadline) throw new Error('timed out waiting for log line')
    await new Promise(r => setTimeout(r, 10))
  }
}

describe('serve HTTP service log', () => {
  let server: http.Server
  afterEach(() => new Promise<void>(r => server.close(() => r())))

  it('logs one wide event per request: mod/method/path/status/ms', async () => {
    const sink = memorySink()
    const { milkie, agentId, broadcaster } = buildMilkie()
    server = createServeServer({
      milkie, agentId, broadcaster,
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    const port = await listen(server)
    expect(await get(port, '/health')).toBe(200)
    const line = await waitFor(() => sink.lines().find(l => l.msg === 'http request'))
    expect(line.mod).toBe('server')
    expect(line.method).toBe('GET')
    expect(line.path).toBe('/health')
    expect(line.status).toBe(200)
    expect(typeof line.ms).toBe('number')
  })

  it('logs 404 and 400 statuses (异常路径同样有 wide event)', async () => {
    const sink = memorySink()
    const { milkie, agentId, broadcaster } = buildMilkie()
    server = createServeServer({
      milkie, agentId, broadcaster,
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    const port = await listen(server)
    await get(port, '/no-such-route')
    await post(port, '/chat', {})                       // 缺 contextId → 400
    const seen = await waitFor(() => {
      const ls = sink.lines().filter(l => l.msg === 'http request')
      return ls.length >= 2 ? ls : undefined
    })
    expect(seen.map(l => [l.path, l.status])).toEqual(expect.arrayContaining([
      ['/no-such-route', 404],
      ['/chat', 400],
    ]))
  })

  it('SSE 请求记录完整流时长（/chat 正常流）', async () => {
    const sink = memorySink()
    const { milkie, agentId, broadcaster } = buildMilkie()
    server = createServeServer({
      milkie, agentId, broadcaster,
      logger: createServiceLogger({ level: 'info', format: 'json', destination: sink.stream }),
    })
    const port = await listen(server)
    const res = await post(port, '/chat', { contextId: 'c1', input: 'hi' })
    expect(res.status).toBe(200)
    expect(res.text).toContain('agent.run.completed')
    const line = await waitFor(() => sink.lines().find(l => l.msg === 'http request' && l.path === '/chat'))
    expect(line.status).toBe(200)
  })
})
