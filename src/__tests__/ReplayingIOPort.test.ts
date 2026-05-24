import { ReplayingIOPort } from '../trace/ReplayingIOPort'
import { CacheIndex } from '../trace/CacheIndex'
import { DefaultIOPort } from '../runtime/IOPort'
import { ReplayDivergenceError } from '../trace/ReplayDivergenceError'
import { hashModelRequest, hashToolCall } from '../trace/hash'
import type { IIOPort } from '../runtime/IOPort'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { Event, LlmRespondedPayload, ToolRespondedPayload } from '../trace/types'

class FailingGateway implements IModelGateway {
  complete(): Promise<ModelResponse> { throw new Error('inner gateway must not be called during replay') }
  async *stream(): AsyncIterable<never> { yield* [] }
}

const innerNeverCalled = (): IIOPort => new DefaultIOPort(new FailingGateway())

const llmResp = (text: string): ModelResponse => ({
  content: [{ type: 'text', text }], toolCalls: [], finishReason: 'end_turn',
})

describe('ReplayingIOPort', () => {
  it('serves LLM from cache; never calls inner', async () => {
    const req: ModelRequest = { model: 'm', messages: [], system: '', tools: [] }
    const h = hashModelRequest(req)
    const ev: Event<LlmRespondedPayload> = {
      id: 'e1', runId: 'r1', type: 'llm.responded', actor: 'runtime', timestamp: 1,
      payload: { response: llmResp('cached'), requestHash: h },
    }
    const port = new ReplayingIOPort(CacheIndex.fromEvents([ev]), innerNeverCalled())
    const resp = await port.invokeLLM(req)
    expect(resp.content[0]).toMatchObject({ text: 'cached' })
  })

  it('throws ReplayDivergenceError on LLM cache miss', async () => {
    const req: ModelRequest = { model: 'm', messages: [], system: '', tools: [] }
    const port = new ReplayingIOPort(CacheIndex.fromEvents([]), innerNeverCalled())
    await expect(port.invokeLLM(req)).rejects.toBeInstanceOf(ReplayDivergenceError)
  })

  it('divergence error carries kind and actualHash', async () => {
    const req: ModelRequest = { model: 'm', messages: [], system: '', tools: [] }
    const port = new ReplayingIOPort(CacheIndex.fromEvents([]), innerNeverCalled())
    try {
      await port.invokeLLM(req)
    } catch (err) {
      const e = err as ReplayDivergenceError
      expect(e.kind).toBe('llm')
      expect(e.actualHash).toBe(hashModelRequest(req))
      expect(e).toBeInstanceOf(ReplayDivergenceError)
    }
  })

  it('serves tool output from cache; execute thunk never runs', async () => {
    const input = { x: 1 }
    const h = hashToolCall('t', input)
    const ev: Event<ToolRespondedPayload> = {
      id: 'e1', runId: 'r1', type: 'tool.responded', actor: 'runtime', timestamp: 1,
      payload: { toolName: 't', output: { ok: true }, requestHash: h },
    }
    const port = new ReplayingIOPort(CacheIndex.fromEvents([ev]), innerNeverCalled())
    let executeCalled = false
    const out = await port.invokeTool('t', input, async () => { executeCalled = true; return 'should not run' })
    expect(out).toEqual({ ok: true })
    expect(executeCalled).toBe(false)
  })

  it('tool error rethrows with retryable preserved', async () => {
    const input = { x: 1 }
    const h = hashToolCall('t', input)
    const ev: Event<ToolRespondedPayload> = {
      id: 'e1', runId: 'r1', type: 'tool.responded', actor: 'runtime', timestamp: 1,
      payload: { toolName: 't', error: { message: 'boom', retryable: true }, requestHash: h },
    }
    const port = new ReplayingIOPort(CacheIndex.fromEvents([ev]), innerNeverCalled())
    try {
      await port.invokeTool('t', input, async () => 'unused')
    } catch (err) {
      const e = err as Error & { retryable?: boolean }
      expect(e.message).toBe('boom')
      expect(e.retryable).toBe(true)
    }
  })

  it('now/uuid passthrough to inner', () => {
    const port = new ReplayingIOPort(CacheIndex.fromEvents([]), new DefaultIOPort(new FailingGateway()))
    expect(typeof port.now()).toBe('number')
    expect(port.uuid()).toMatch(/^[0-9a-f-]{36}$/i)
  })
})
