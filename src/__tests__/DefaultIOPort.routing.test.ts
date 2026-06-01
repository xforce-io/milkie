import { DefaultIOPort } from '../runtime/IOPort'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'

class SpyGateway implements IModelGateway {
  completeCalled = false
  streamCalled = false
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.completeCalled = true
    return { content: [{ type: 'text', text: 'from-complete' }], toolCalls: [] }
  }
  async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
    this.streamCalled = true
    yield { type: 'message_delta', data: { text: 'from-' } }
    yield { type: 'message_delta', data: { text: 'stream' } }
  }
}

class ToolCallGateway implements IModelGateway {
  completeCalled = false
  streamCalled = false
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.completeCalled = true
    return { content: [], toolCalls: [] }
  }
  async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
    this.streamCalled = true
    yield { type: 'tool_call_start', data: { toolCallId: 'tc-1', name: 'search' } }
    yield { type: 'tool_call_delta', data: { toolCallId: 'tc-1', delta: '{"q":' } }
    yield { type: 'tool_call_delta', data: { toolCallId: 'tc-1', delta: '"milkie"}' } }
    yield { type: 'tool_call_done', data: { toolCallId: 'tc-1', input: undefined } }
  }
}

const req: ModelRequest = { model: 'm', messages: [] }

describe('DefaultIOPort.invokeLLM routing', () => {
  it('无 onEvent → 走 complete()，不调用 stream()，返回 complete 内容', async () => {
    const gw = new SpyGateway()
    const port = new DefaultIOPort(gw)

    const res = await port.invokeLLM(req)

    expect(gw.completeCalled).toBe(true)
    expect(gw.streamCalled).toBe(false)
    expect(res.content).toEqual([{ type: 'text', text: 'from-complete' }])
    expect(res.toolCalls).toEqual([])
  })

  it('有 onEvent → 走 stream()，不调用 complete()，返回聚合内容且 onEvent 收到全部事件', async () => {
    const gw = new SpyGateway()
    const port = new DefaultIOPort(gw)
    const events: ModelEvent[] = []

    const res = await port.invokeLLM(req, (e) => events.push(e))

    expect(gw.streamCalled).toBe(true)
    expect(gw.completeCalled).toBe(false)
    expect(res.content).toEqual([{ type: 'text', text: 'from-stream' }])
    expect(events).toHaveLength(2)
    expect(events).toEqual([
      { type: 'message_delta', data: { text: 'from-' } },
      { type: 'message_delta', data: { text: 'stream' } },
    ])
  })

  it('有 onEvent 且流含 tool_call → 端到端经 aggregateStream 聚合出正确 toolCalls', async () => {
    const gw = new ToolCallGateway()
    const port = new DefaultIOPort(gw)
    const events: ModelEvent[] = []

    const res = await port.invokeLLM(req, (e) => events.push(e))

    expect(gw.streamCalled).toBe(true)
    expect(gw.completeCalled).toBe(false)
    expect(res.toolCalls).toEqual([
      { id: 'tc-1', name: 'search', input: { q: 'milkie' } },
    ])
    expect(res.content).toEqual([
      { type: 'tool_use', id: 'tc-1', name: 'search', input: { q: 'milkie' } },
    ])
    expect(events).toHaveLength(4)
  })

  it('显式传 onEvent=undefined → 仍走 complete()', async () => {
    const gw = new SpyGateway()
    const port = new DefaultIOPort(gw)

    const res = await port.invokeLLM(req, undefined)

    expect(gw.completeCalled).toBe(true)
    expect(gw.streamCalled).toBe(false)
    expect(res.content).toEqual([{ type: 'text', text: 'from-complete' }])
  })
})
