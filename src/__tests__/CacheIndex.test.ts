import { CacheIndex } from '../trace/CacheIndex'
import type { Event, LlmRespondedPayload, ToolRespondedPayload } from '../trace/types'
import type { ModelResponse } from '../types/model'

const llmResp = (text: string): ModelResponse => ({
  content:      [{ type: 'text', text }],
  toolCalls:    [],
  finishReason: 'end_turn',
})

const mkLlmResponded = (hash: string, text: string): Event<LlmRespondedPayload> => ({
  id:        `e-${Math.random()}`,
  runId:     'r1',
  type:      'llm.responded',
  actor:     'runtime',
  timestamp: 1,
  payload:   { response: llmResp(text), requestHash: hash },
})

const mkToolResponded = (hash: string, output?: unknown, error?: NonNullable<ToolRespondedPayload['error']>): Event<ToolRespondedPayload> => ({
  id:        `e-${Math.random()}`,
  runId:     'r1',
  type:      'tool.responded',
  actor:     'runtime',
  timestamp: 1,
  payload:   { toolName: 't', output, error, requestHash: hash },
})

describe('CacheIndex', () => {
  it('fromEvents builds empty index for empty events', () => {
    const idx = CacheIndex.fromEvents([])
    expect(idx.remaining()).toEqual({ llm: 0, tool: 0, clock: 0, uuid: 0 })
  })

  it('consumeLLM serves cached responses in FIFO order per hash', () => {
    const idx = CacheIndex.fromEvents([
      mkLlmResponded('h1', 'first'),
      mkLlmResponded('h1', 'second'),
      mkLlmResponded('h2', 'other'),
    ])
    expect(idx.consumeLLM('h1').content[0]).toMatchObject({ text: 'first' })
    expect(idx.consumeLLM('h1').content[0]).toMatchObject({ text: 'second' })
    expect(idx.consumeLLM('h2').content[0]).toMatchObject({ text: 'other' })
    expect(idx.remaining()).toEqual({ llm: 0, tool: 0, clock: 0, uuid: 0 })
  })

  it('consumeLLM throws when queue exhausted', () => {
    const idx = CacheIndex.fromEvents([mkLlmResponded('h1', 'x')])
    idx.consumeLLM('h1')
    expect(() => idx.consumeLLM('h1')).toThrow(/queue empty/)
  })

  it('consumeLLM throws when hash never seen', () => {
    const idx = CacheIndex.fromEvents([])
    expect(() => idx.consumeLLM('nope')).toThrow(/queue empty/)
  })

  it('consumeTool returns output for successful tool', () => {
    const idx = CacheIndex.fromEvents([mkToolResponded('h1', { ok: true })])
    expect(idx.consumeTool('h1')).toEqual({ ok: true })
  })

  it('consumeTool rethrows Error with retryable/code/name preserved', () => {
    const idx = CacheIndex.fromEvents([
      mkToolResponded('h1', undefined, { message: 'boom', retryable: true, code: 'EBUSY', name: 'BusyError' }),
    ])
    try {
      idx.consumeTool('h1')
      throw new Error('expected throw')
    } catch (err) {
      const e = err as Error & { retryable?: boolean; code?: string }
      expect(e.message).toBe('boom')
      expect(e.retryable).toBe(true)
      expect(e.code).toBe('EBUSY')
      expect(e.name).toBe('BusyError')
    }
  })

  it('remaining tracks unconsumed counts', () => {
    const idx = CacheIndex.fromEvents([
      mkLlmResponded('h1', 'a'),
      mkLlmResponded('h1', 'b'),
      mkToolResponded('h2', 'out'),
    ])
    expect(idx.remaining()).toEqual({ llm: 2, tool: 1, clock: 0, uuid: 0 })
    idx.consumeLLM('h1')
    expect(idx.remaining()).toEqual({ llm: 1, tool: 1, clock: 0, uuid: 0 })
  })

  it('allHashes returns all unique hashes seen', () => {
    const idx = CacheIndex.fromEvents([
      mkLlmResponded('h1', 'a'),
      mkLlmResponded('h1', 'b'),
      mkLlmResponded('h2', 'c'),
      mkToolResponded('th1', 'out'),
    ])
    expect(idx.allHashes().llm.sort()).toEqual(['h1', 'h2'])
    expect(idx.allHashes().tool).toEqual(['th1'])
  })
})
