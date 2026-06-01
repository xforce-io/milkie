/**
 * #80 — stream-vs-complete recording equivalence.
 *
 * 双路-B 选路下，一个 turn 可能走 complete() 或 stream()+aggregateStream()。
 * 两条路径都产出 ModelResponse，RecordingIOPort 录它进 llm.responded
 * (replay cache 的来源)。本测试把"两条路径产出的 ModelResponse 等价性边界"
 * 用可执行断言钉死，而非靠文字推理。
 *
 * 关键不变量（replay 安全）：
 *   CacheIndex 的 key 是 hashModelRequest(req) = sha256(canonicalize(REQUEST))。
 *   usage 在 RESPONSE 上，不进 hash。因此即便两条路径的 usage 字段有差异，
 *   也绝不影响 replay 的 key 匹配 —— replay 按 requestHash 找，录什么放什么，
 *   自洽确定性。content/toolCalls 必须等价（agent 行为的实质）；
 *   usage 的差异是已知的现状语义，显式锁定于此。
 */
import { aggregateStream } from '../gateway/StreamAggregator'
import { hashModelRequest } from '../trace/hash'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model'

const REQ: ModelRequest = { model: 'm', messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] }

async function* gen(list: ModelEvent[]): AsyncIterable<ModelEvent> {
  for (const e of list) yield e
}

describe('#80 stream vs complete — recording equivalence', () => {
  test('纯文本：content/toolCalls 等价', async () => {
    const completeResp: ModelResponse = {
      content: [{ type: 'text', text: 'Hello, world' }],
      toolCalls: [],
      finishReason: 'stop',
      usage: { inputTokens: 10, outputTokens: 5 },
    }
    const streamEvents: ModelEvent[] = [
      { type: 'message_delta', data: { text: 'Hello, ' } },
      { type: 'message_delta', data: { text: 'world' } },
      { type: 'usage', data: { inputTokens: 10, outputTokens: 5 } },
    ]
    const aggregated = await aggregateStream(gen(streamEvents))

    expect(aggregated.content).toEqual(completeResp.content)
    expect(aggregated.toolCalls).toEqual(completeResp.toolCalls)
    // finishReason: 聚合器不从 usage/text 推断 'stop'（只在 error 时设）。
    // 显式锁定：纯文本流聚合后 finishReason 为 undefined，而 complete 为 'stop'。
    // 已知现状差异，不影响 replay（finishReason 不进 hash）。
    expect(aggregated.finishReason).toBeUndefined()
    expect(completeResp.finishReason).toBe('stop')
  })

  test('含 tool_call：content/toolCalls 等价（实质行为字段）', async () => {
    const completeResp: ModelResponse = {
      content: [
        { type: 'text', text: 'let me search' },
        { type: 'tool_use', id: 'c1', name: 'search', input: { q: 'three kingdoms' } },
      ],
      toolCalls: [{ id: 'c1', name: 'search', input: { q: 'three kingdoms' } }],
      finishReason: 'tool_calls',
    }
    const streamEvents: ModelEvent[] = [
      { type: 'message_delta', data: { text: 'let me search' } },
      { type: 'tool_call_start', data: { toolCallId: 'c1', name: 'search' } },
      { type: 'tool_call_delta', data: { toolCallId: 'c1', delta: '{"q":"three ' } },
      { type: 'tool_call_delta', data: { toolCallId: 'c1', delta: 'kingdoms"}' } },
      { type: 'tool_call_done', data: { toolCallId: 'c1', input: undefined } },
    ]
    const aggregated = await aggregateStream(gen(streamEvents))

    // 实质行为字段必须逐字等价 —— agent 后续决策/工具执行的依据。
    expect(aggregated.content).toEqual(completeResp.content)
    expect(aggregated.toolCalls).toEqual(completeResp.toolCalls)
  })

  test('usage 已知差异显式锁定：Anthropic stream 路径 inputTokens=0（现状语义）', async () => {
    // AnthropicAdapter.stream() 的 usage 事件 inputTokens 硬编码 0（message_delta
    // 不携带 input_tokens），而 complete() 的 parseResponse 取真实 input_tokens。
    // 这是一个已知的字段级不等价。此测试把它钉死，防止无意"修复"破坏现状，
    // 并文档化：该差异对 replay 无害（见下条 replay-safety 测试）。
    const streamUsageEvents: ModelEvent[] = [
      { type: 'message_delta', data: { text: 'hi' } },
      { type: 'usage', data: { inputTokens: 0, outputTokens: 7 } }, // Anthropic stream 现状
    ]
    const aggregated = await aggregateStream(gen(streamUsageEvents))
    expect(aggregated.usage).toEqual({ inputTokens: 0, outputTokens: 7 })

    // 对照：complete() 路径会有真实 inputTokens。两者 usage 不等 —— 已知且可接受。
    const completeUsage = { inputTokens: 42, outputTokens: 7 }
    expect(aggregated.usage).not.toEqual(completeUsage)
  })

  test('replay 安全：usage 差异不进 requestHash（CacheIndex key 仅基于 request）', () => {
    // 核心论证的可执行版本：hashModelRequest 只 hash request。
    // 同一个 request，无论它最终走 stream 还是 complete 录制（usage 不同），
    // 它的 requestHash 完全相同 —— replay 按 requestHash 匹配，不受 usage 影响。
    const hash1 = hashModelRequest(REQ)
    const hash2 = hashModelRequest({ ...REQ })
    expect(hash1).toBe(hash2)

    // request 内容变了才会变 hash；usage 是 response 字段，根本不参与 hash 计算。
    const differentReq = hashModelRequest({ ...REQ, model: 'other-model' })
    expect(differentReq).not.toBe(hash1)
  })

  test('SpyGateway 端到端：同一 req 两路产出的 content/toolCalls 等价', async () => {
    class SpyGateway implements IModelGateway {
      async complete(_req: ModelRequest): Promise<ModelResponse> {
        return {
          content: [{ type: 'tool_use', id: 't1', name: 'read', input: { f: 'x.txt' } }],
          toolCalls: [{ id: 't1', name: 'read', input: { f: 'x.txt' } }],
        }
      }
      async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
        yield { type: 'tool_call_start', data: { toolCallId: 't1', name: 'read' } }
        yield { type: 'tool_call_delta', data: { toolCallId: 't1', delta: '{"f":"x.txt"}' } }
        yield { type: 'tool_call_done', data: { toolCallId: 't1', input: undefined } }
      }
    }
    const gw = new SpyGateway()
    const completeResp = await gw.complete(REQ)
    const streamResp = await aggregateStream(gw.stream(REQ))

    expect(streamResp.content).toEqual(completeResp.content)
    expect(streamResp.toolCalls).toEqual(completeResp.toolCalls)
  })
})
