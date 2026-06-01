import { RecordingIOPort } from '../trace/RecordingIOPort'
import { DefaultIOPort } from '../runtime/IOPort'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { LlmRespondedPayload } from '../trace/types'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'

/**
 * 验证 RecordingIOPort 把 onEvent 透传给 inner（录制路径也能流式），
 * 但录制逻辑完全不变：录的仍是 DefaultIOPort 经 aggregateStream 聚合好的
 * 完整 ModelResponse；delta 只通过 onEvent 出去，不进 EventStore。
 */

const REQ: ModelRequest = { provider: 'stub', model: 'stub', messages: [] } as ModelRequest

class StreamGateway implements IModelGateway {
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    return { content: [{ type: 'text', text: 'X' }], toolCalls: [] }
  }
  async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
    yield { type: 'message_delta', data: { text: 'he' } }
    yield { type: 'message_delta', data: { text: 'llo' } }
  }
}

class ToolStreamGateway implements IModelGateway {
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    return { content: [{ type: 'text', text: 'X' }], toolCalls: [] }
  }
  async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
    yield { type: 'tool_call_start', data: { toolCallId: 't1', name: 'search' } }
    yield { type: 'tool_call_delta', data: { toolCallId: 't1', delta: '{"q":' } }
    yield { type: 'tool_call_delta', data: { toolCallId: 't1', delta: '"hi"}' } }
    yield { type: 'tool_call_done', data: { toolCallId: 't1', input: { q: 'hi' } } }
  }
}

function llmResponded(events: { type: string; payload: unknown }[]): LlmRespondedPayload[] {
  return events.filter(e => e.type === 'llm.responded').map(e => e.payload as LlmRespondedPayload)
}

describe('RecordingIOPort — onEvent 透传 + 录制不变', () => {
  it('透传 onEvent，且录完整聚合后的 llm.responded', async () => {
    const inner = new DefaultIOPort(new StreamGateway())
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r1')

    const seen: ModelEvent[] = []
    const response = await port.invokeLLM(REQ, e => seen.push(e))

    // onEvent 收到 2 个 delta（流式透传到 inner 的证据）
    expect(seen).toEqual([
      { type: 'message_delta', data: { text: 'he' } },
      { type: 'message_delta', data: { text: 'llo' } },
    ])

    // 返回的是聚合后的完整 response
    expect(response.content).toEqual([{ type: 'text', text: 'hello' }])
    expect(response.toolCalls).toEqual([])

    // 录制不变：store 有 llm.responded，且 payload.response 是聚合后的完整 response
    const events = await store.readByRunId('r1')
    const responded = llmResponded(events)
    expect(responded).toHaveLength(1)
    expect(responded[0]!.response.content).toEqual([{ type: 'text', text: 'hello' }])
    // delta 不进 EventStore：没有任何 message_delta 类型的事件
    expect(events.some(e => (e.type as string) === 'message_delta')).toBe(false)
  })

  it('无 onEvent 时走 complete，且仍照常录', async () => {
    const inner = new DefaultIOPort(new StreamGateway())
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r2')

    let called = 0
    // 不传 onEvent
    const response = await port.invokeLLM(REQ)
    expect(called).toBe(0)
    void called

    // 走 complete 路径 → {text:'X'}
    expect(response.content).toEqual([{ type: 'text', text: 'X' }])

    const events = await store.readByRunId('r2')
    const responded = llmResponded(events)
    expect(responded).toHaveLength(1)
    expect(responded[0]!.response.content).toEqual([{ type: 'text', text: 'X' }])
  })

  it('tool_call 流经 Recording 时录制的是聚合后完整 toolCalls', async () => {
    const inner = new DefaultIOPort(new ToolStreamGateway())
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r3')

    const seen: ModelEvent[] = []
    const response = await port.invokeLLM(REQ, e => seen.push(e))

    // 流式事件透传（4 个 tool_call 事件）
    expect(seen.map(e => e.type)).toEqual([
      'tool_call_start', 'tool_call_delta', 'tool_call_delta', 'tool_call_done',
    ])

    // 返回聚合后的完整 toolCalls
    expect(response.toolCalls).toEqual([{ id: 't1', name: 'search', input: { q: 'hi' } }])

    // 录制的 llm.responded 携带完整 toolCalls（replay cache 完整的证据）
    const events = await store.readByRunId('r3')
    const responded = llmResponded(events)
    expect(responded).toHaveLength(1)
    expect(responded[0]!.response.toolCalls).toEqual([
      { id: 't1', name: 'search', input: { q: 'hi' } },
    ])
    expect(responded[0]!.response.content).toEqual([
      { type: 'tool_use', id: 't1', name: 'search', input: { q: 'hi' } },
    ])
  })

  it('llm.requested 也照常录（录制流程未被破坏）', async () => {
    const inner = new DefaultIOPort(new StreamGateway())
    const store = new MemoryEventStore()
    const port  = new RecordingIOPort(inner, store, 'r4')

    await port.invokeLLM(REQ, () => {})

    const events = await store.readByRunId('r4')
    const requested = events.filter(e => e.type === 'llm.requested')
    expect(requested).toHaveLength(1)
    // requested 在 responded 之前
    const reqIdx  = events.findIndex(e => e.type === 'llm.requested')
    const respIdx = events.findIndex(e => e.type === 'llm.responded')
    expect(reqIdx).toBeGreaterThan(-1)
    expect(respIdx).toBeGreaterThan(reqIdx)
  })
})
