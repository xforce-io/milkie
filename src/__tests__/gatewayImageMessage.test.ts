// #236: image message contract + gateway capability/format boundaries.
import { OpenAICompatibleAdapter } from '../gateway/OpenAICompatibleAdapter'
import { AnthropicAdapter } from '../gateway/AnthropicAdapter'
import { ModelGatewayError } from '../gateway/ModelGatewayError'
import { DefaultIOPort } from '../runtime/IOPort'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { RecordingIOPort } from '../trace/RecordingIOPort'
import { sanitizeModelRequestForTrace, redactImageUrl } from '../trace/imageSummary'
import type { ModelRequest, ModelResponse, IModelGateway } from '../types/model'
import type { Message } from '../types/common'
import { createHash } from 'crypto'

const PNG_1X1_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='

function textImageMessages(): Message[] {
  return [{
    role: 'user',
    content: [
      { type: 'text', text: 'what is in' },
      {
        type: 'image',
        mediaType: 'image/png',
        source: { kind: 'base64', data: PNG_1X1_BASE64 },
      },
      { type: 'text', text: 'this image?' },
    ],
  }]
}

function visionRequest(over: Partial<ModelRequest> = {}): ModelRequest {
  return {
    model: 'vision-model',
    messages: textImageMessages(),
    ...over,
  }
}

// Access private convert / build helpers the same way existing adapter tests do.
function openaiMessages(adapter: OpenAICompatibleAdapter, req: ModelRequest): unknown[] {
  return (adapter as unknown as { convertMessages(r: ModelRequest): unknown[] }).convertMessages(req)
}
function anthropicParams(adapter: AnthropicAdapter, req: ModelRequest): Record<string, unknown> {
  return (adapter as unknown as { buildParams(r: ModelRequest): Record<string, unknown> }).buildParams(req)
}

describe('#236 gateway image message contract', () => {
  describe('OpenAI-compatible adapter', () => {
    it('S1: maps ordered text+image to image_url parts when imageInput is enabled', async () => {
      const adapter = new OpenAICompatibleAdapter({
        apiKey: 'sk-test',
        capabilities: { imageInput: true },
      })
      const calls: unknown[] = []
      ;(adapter as unknown as { client: { chat: { completions: { create: unknown } } } })
        .client.chat.completions.create = async (params: unknown) => {
          calls.push(params)
          return {
            choices: [{
              message: {
                role: 'assistant',
                content: null,
                tool_calls: [{
                  id: 'call-1',
                  type: 'function',
                  function: { name: 'look', arguments: '{}' },
                }],
              },
              finish_reason: 'tool_calls',
            }],
            usage: { prompt_tokens: 1, completion_tokens: 1 },
          }
        }

      const res = await adapter.complete(visionRequest({
        tools: [{ name: 'look', description: 'd', inputSchema: {} }],
      }))
      expect(res.toolCalls).toEqual([
        expect.objectContaining({ name: 'look' }),
      ])

      const params = calls[0] as { messages: Array<{ role: string; content: unknown }> }
      const user = params.messages.find(m => m.role === 'user')!
      expect(Array.isArray(user.content)).toBe(true)
      const parts = user.content as Array<{ type: string; text?: string; image_url?: { url: string } }>
      expect(parts.map(p => p.type)).toEqual(['text', 'image_url', 'text'])
      expect(parts[0]).toMatchObject({ type: 'text', text: 'what is in' })
      expect(parts[1]!.image_url!.url).toBe(`data:image/png;base64,${PNG_1X1_BASE64}`)
      expect(parts[2]).toMatchObject({ type: 'text', text: 'this image?' })
    })

    it('S2: no imageInput capability fails before network with MODEL_CAPABILITY_UNSUPPORTED', async () => {
      const adapter = new OpenAICompatibleAdapter({ apiKey: 'sk-test' }) // default false
      let networked = false
      ;(adapter as unknown as { client: { chat: { completions: { create: unknown } } } })
        .client.chat.completions.create = async () => {
          networked = true
          return { choices: [{ message: { role: 'assistant', content: 'x' }, finish_reason: 'stop' }] }
        }

      await expect(adapter.complete(visionRequest())).rejects.toMatchObject({
        name: 'ModelGatewayError',
        envelope: {
          code: 'MODEL_CAPABILITY_UNSUPPORTED',
          phase: 'request',
          retryable: false,
          capability: 'imageInput',
        },
      })
      expect(networked).toBe(false)
      // Pure text still works without capability.
      await expect(adapter.complete({
        model: 'm',
        messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      })).resolves.toBeDefined()
    })

    it('rejects http URL, invalid base64, and unsupported media type before network', async () => {
      const adapter = new OpenAICompatibleAdapter({
        apiKey: 'sk-test',
        capabilities: { imageInput: true },
      })
      let networked = false
      ;(adapter as unknown as { client: { chat: { completions: { create: unknown } } } })
        .client.chat.completions.create = async () => {
          networked = true
          return { choices: [{ message: { role: 'assistant', content: 'x' }, finish_reason: 'stop' }] }
        }

      await expect(adapter.complete({
        model: 'm',
        messages: [{
          role: 'user',
          content: [{
            type: 'image',
            mediaType: 'image/png',
            source: { kind: 'url', url: 'http://example.com/a.png' },
          }],
        }],
      })).rejects.toMatchObject({ envelope: { code: 'MODEL_BAD_RESPONSE', phase: 'request' } })

      await expect(adapter.complete({
        model: 'm',
        messages: [{
          role: 'user',
          content: [{
            type: 'image',
            mediaType: 'image/png',
            source: { kind: 'base64', data: '%%%not-base64%%%' },
          }],
        }],
      })).rejects.toMatchObject({ envelope: { code: 'MODEL_BAD_RESPONSE', phase: 'request' } })

      await expect(adapter.complete({
        model: 'm',
        messages: [{
          role: 'user',
          content: [{
            type: 'image',
            mediaType: 'image/tiff' as 'image/png',
            source: { kind: 'base64', data: PNG_1X1_BASE64 },
          }],
        }],
      })).rejects.toMatchObject({ envelope: { code: 'MODEL_BAD_RESPONSE', phase: 'request' } })

      expect(networked).toBe(false)
    })

    it('rejects non-canonical base64 padding/length before network (AA= and kin)', async () => {
      // Node Buffer.from is lenient (e.g. AA= decodes) but #236 requires strict
      // standard padded base64: invalid padding/length must fail closed pre-network.
      const adapter = new OpenAICompatibleAdapter({
        apiKey: 'sk-test',
        capabilities: { imageInput: true },
      })
      let networked = false
      ;(adapter as unknown as { client: { chat: { completions: { create: unknown } } } })
        .client.chat.completions.create = async () => {
          networked = true
          return { choices: [{ message: { role: 'assistant', content: 'x' }, finish_reason: 'stop' }] }
        }

      for (const data of ['AA=', 'A===', 'A', 'AAA', 'AA==x', '=AA=', 'AA=_', 'AAAA====']) {
        networked = false
        await expect(adapter.complete({
          model: 'm',
          messages: [{
            role: 'user',
            content: [{
              type: 'image',
              mediaType: 'image/png',
              source: { kind: 'base64', data },
            }],
          }],
        })).rejects.toMatchObject({
          name: 'ModelGatewayError',
          envelope: { code: 'MODEL_BAD_RESPONSE', phase: 'request', retryable: false },
        })
        expect(networked).toBe(false)
      }

      // Canonical 1x1 fixture still accepted (network may run).
      networked = false
      await expect(adapter.complete(visionRequest({ model: 'm' }))).resolves.toBeDefined()
      expect(networked).toBe(true)
    })



    it('capabilities are never inferred from provider/endpoint names', () => {
      const a = new OpenAICompatibleAdapter({
        provider: 'openai',
        baseUrl: 'https://api.openai.com/v1',
        apiKey: 'sk',
      })
      expect(a.capabilities.imageInput).toBe(false)
    })

    it('maps assistant-role image parts in order (no silent drop)', () => {
      const adapter = new OpenAICompatibleAdapter({
        apiKey: 'sk-test',
        capabilities: { imageInput: true },
      })
      const req: ModelRequest = {
        model: 'vision-model',
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'user caption' },
              {
                type: 'image',
                mediaType: 'image/png',
                source: { kind: 'url', url: 'https://cdn.example.com/user.png' },
              },
            ],
          },
          {
            role: 'assistant',
            content: [
              { type: 'text', text: 'assistant saw' },
              {
                type: 'image',
                mediaType: 'image/jpeg',
                source: { kind: 'base64', data: PNG_1X1_BASE64 },
              },
              { type: 'text', text: 'this frame' },
            ],
          },
        ],
      }
      const msgs = openaiMessages(adapter, req) as Array<{
        role: string
        content: unknown
      }>
      const user = msgs.find(m => m.role === 'user')!
      const assistant = msgs.find(m => m.role === 'assistant')!

      expect(Array.isArray(user.content)).toBe(true)
      const userParts = user.content as Array<{ type: string; text?: string; image_url?: { url: string } }>
      expect(userParts.map(p => p.type)).toEqual(['text', 'image_url'])
      expect(userParts[0]).toMatchObject({ type: 'text', text: 'user caption' })
      expect(userParts[1]!.image_url!.url).toBe('https://cdn.example.com/user.png')

      // Assistant must not silently drop images. Wire must keep order.
      expect(Array.isArray(assistant.content)).toBe(true)
      const asstParts = assistant.content as Array<{ type: string; text?: string; image_url?: { url: string } }>
      expect(asstParts.map(p => p.type)).toEqual(['text', 'image_url', 'text'])
      expect(asstParts[0]).toMatchObject({ type: 'text', text: 'assistant saw' })
      expect(asstParts[1]!.image_url!.url).toBe(`data:image/jpeg;base64,${PNG_1X1_BASE64}`)
      expect(asstParts[2]).toMatchObject({ type: 'text', text: 'this frame' })
    })

  })

  describe('Anthropic adapter', () => {
    it('S0: default (no capabilities) rejects image before network', async () => {
      // #236 fail-closed: undeclared imageInput must not optimistically accept vision.
      const adapter = new AnthropicAdapter({ apiKey: 'sk-test' })
      expect(adapter.capabilities.imageInput).toBe(false)
      let networked = false
      ;(adapter as unknown as { client: { messages: { create: unknown } } })
        .client.messages.create = async () => {
          networked = true
          return {
            id: 'msg_x',
            type: 'message',
            role: 'assistant',
            model: 'claude',
            content: [{ type: 'text', text: 'x' }],
            stop_reason: 'end_turn',
            usage: { input_tokens: 1, output_tokens: 1 },
          }
        }
      await expect(adapter.complete(visionRequest({ model: 'claude' }))).rejects.toMatchObject({
        envelope: {
          code: 'MODEL_CAPABILITY_UNSUPPORTED',
          phase: 'request',
          retryable: false,
          capability: 'imageInput',
        },
      })
      expect(networked).toBe(false)
    })

    it('S1: maps ordered text+image to Anthropic image sources via complete()', async () => {
      const adapter = new AnthropicAdapter({
        apiKey: 'sk-test',
        capabilities: { imageInput: true },
      })
      const calls: unknown[] = []
      ;(adapter as unknown as { client: { messages: { create: unknown } } })
        .client.messages.create = async (params: unknown) => {
          calls.push(params)
          return {
            id: 'msg_1',
            type: 'message',
            role: 'assistant',
            model: 'claude-sonnet-4-6',
            content: [{
              type: 'tool_use',
              id: 'toolu_1',
              name: 'look',
              input: {},
            }],
            stop_reason: 'tool_use',
            usage: { input_tokens: 1, output_tokens: 1 },
          }
        }

      const res = await adapter.complete(visionRequest({
        model: 'claude-sonnet-4-6',
        tools: [{ name: 'look', description: 'd', inputSchema: {} }],
      }))
      expect(res.toolCalls).toEqual([
        expect.objectContaining({ id: 'toolu_1', name: 'look' }),
      ])

      const params = calls[0] as {
        stream?: boolean
        messages: Array<{ role: string; content: Array<Record<string, unknown>> }>
      }
      expect(params.stream).toBe(false)
      const content = params.messages[0]!.content
      expect(content.map(c => c.type)).toEqual(['text', 'image', 'text'])
      expect(content[0]).toMatchObject({ type: 'text', text: 'what is in' })
      expect(content[1]).toMatchObject({
        type: 'image',
        source: {
          type: 'base64',
          media_type: 'image/png',
          data: PNG_1X1_BASE64,
        },
      })
      expect(content[2]).toMatchObject({ type: 'text', text: 'this image?' })
    })

    it('S2: capabilities.imageInput=false rejects before network', async () => {
      const adapter = new AnthropicAdapter({
        apiKey: 'sk-test',
        capabilities: { imageInput: false },
      })
      let networked = false
      ;(adapter as unknown as { client: { messages: { create: unknown } } })
        .client.messages.create = async () => {
          networked = true
          return {
            id: 'msg_x',
            type: 'message',
            role: 'assistant',
            model: 'claude',
            content: [{ type: 'text', text: 'x' }],
            stop_reason: 'end_turn',
            usage: { input_tokens: 1, output_tokens: 1 },
          }
        }
      await expect(adapter.complete(visionRequest({ model: 'claude' }))).rejects.toMatchObject({
        envelope: {
          code: 'MODEL_CAPABILITY_UNSUPPORTED',
          phase: 'request',
          retryable: false,
          capability: 'imageInput',
        },
      })
      expect(networked).toBe(false)
    })

    it('maps https URL image sources when imageInput is explicit true', () => {
      const adapter = new AnthropicAdapter({
        apiKey: 'sk-test',
        capabilities: { imageInput: true },
      })
      const params = anthropicParams(adapter, {
        model: 'claude',
        messages: [{
          role: 'user',
          content: [{
            type: 'image',
            mediaType: 'image/jpeg',
            source: { kind: 'url', url: 'https://cdn.example.com/a.jpg' },
          }],
        }],
      })
      const content = (params['messages'] as Array<{ content: unknown[] }>)[0]!.content
      expect(content[0]).toEqual({
        type: 'image',
        source: { type: 'url', url: 'https://cdn.example.com/a.jpg' },
      })
    })

    it('text-only complete still works without image capabilities', async () => {
      const adapter = new AnthropicAdapter({ apiKey: 'sk-test' })
      expect(adapter.capabilities.imageInput).toBe(false)
      let networked = false
      ;(adapter as unknown as { client: { messages: { create: unknown } } })
        .client.messages.create = async (params: unknown) => {
          networked = true
          return {
            id: 'msg_t',
            type: 'message',
            role: 'assistant',
            model: 'claude',
            content: [{ type: 'text', text: 'hello' }],
            stop_reason: 'end_turn',
            usage: { input_tokens: 1, output_tokens: 1 },
          }
        }
      const res = await adapter.complete({
        model: 'claude',
        messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      })
      expect(networked).toBe(true)
      expect(res.content).toEqual([{ type: 'text', text: 'hello' }])
    })
  })

  describe('trace / record safety', () => {
    it('redacts URL userinfo, query, and fragment; keeps provider input intact', async () => {
      const sensitive =
        'https://user:password@cdn.example.com/a.png?token=secret#frag'
      const redacted = redactImageUrl(sensitive)
      expect(redacted).toBe('https://cdn.example.com/a.png')
      expect(redacted).not.toContain('user:password@')
      expect(redacted).not.toContain('password@')
      expect(redacted).not.toContain('token=secret')
      expect(redacted).not.toContain('#frag')
      expect(redactImageUrl('https://cdn.example.com/a.png?token=secret#frag'))
        .toBe('https://cdn.example.com/a.png')

      // Provider / live gateway must still see the original credentialed URL.
      // Only the trace representation is redacted.
      const sensitiveReq: ModelRequest = {
        model: 'vision-model',
        messages: [{
          role: 'user',
          content: [{
            type: 'image',
            mediaType: 'image/png',
            source: { kind: 'url', url: sensitive },
          }],
        }],
      }
      const safeUrlReq = sanitizeModelRequestForTrace(sensitiveReq)
      const safeUrlImg = safeUrlReq.messages[0]!.content[0] as {
        source: { kind: string; url?: string }
      }
      expect(safeUrlImg.source.url).toBe('https://cdn.example.com/a.png')
      expect(JSON.stringify(safeUrlReq)).not.toContain('user:password@')
      expect(JSON.stringify(safeUrlReq)).not.toContain('password@')
      expect(JSON.stringify(safeUrlReq)).not.toContain('token=secret')
      expect(JSON.stringify(safeUrlReq)).not.toContain('#frag')

      const req = visionRequest()
      const safe = sanitizeModelRequestForTrace(req)
      const img = safe.messages[0]!.content.find(c => (c as { type: string }).type === 'image') as {
        type: 'image'
        mediaType: string
        source: { kind: string; sha256?: string; byteLength?: number; data?: string; url?: string }
      }
      expect(img.source.kind).toBe('base64')
      expect(img.source.data).toBeUndefined()
      expect(img.source.sha256).toBe(
        createHash('sha256').update(Buffer.from(PNG_1X1_BASE64, 'base64')).digest('hex'),
      )
      expect(img.source.byteLength).toBe(Buffer.from(PNG_1X1_BASE64, 'base64').length)

      // RecordingIOPort persists sanitized request while hashing/passing the live one.
      const store = new MemoryEventStore()
      let seenByGateway: ModelRequest | undefined
      class StubGw implements IModelGateway {
        async complete(request: ModelRequest): Promise<ModelResponse> {
          seenByGateway = request
          return { content: [{ type: 'text', text: 'ok' }], toolCalls: [], finishReason: 'end_turn' }
        }
        async *stream(): AsyncIterable<never> { yield* [] }
        readonly capabilities = { imageInput: true }
      }
      // Bypass adapter validation — unit under test is RecordingIOPort sanitization.
      const port = new RecordingIOPort(new DefaultIOPort(new StubGw()), store, 'r-img')
      await port.invokeLLM(sensitiveReq)
      expect(seenByGateway).toBeDefined()
      const liveImg = seenByGateway!.messages[0]!.content[0] as {
        source: { kind: string; url?: string }
      }
      expect(liveImg.source.url).toBe(sensitive)

      const events = await store.readByRunId('r-img')
      const requested = events.find(e => e.type === 'llm.requested')!
      const payloadReq = (requested.payload as { request: ModelRequest }).request
      const serialized = JSON.stringify(payloadReq)
      expect(serialized).not.toContain('user:password')
      expect(serialized).not.toContain('password')
      expect(serialized).not.toContain('token=secret')
      expect(serialized).not.toContain('#frag')
      expect(serialized).toContain('https://cdn.example.com/a.png')

      // Base64 payload must also stay out of the event log.
      await port.invokeLLM(req)
      const events2 = await store.readByRunId('r-img')
      const requested2 = events2.filter(e => e.type === 'llm.requested').at(-1)!
      const serialized2 = JSON.stringify((requested2.payload as { request: ModelRequest }).request)
      expect(serialized2).not.toContain(PNG_1X1_BASE64)
      expect(serialized2).toContain('sha256')
    })
  })

  describe('undeclared custom gateway', () => {
    it('DefaultIOPort fails closed before complete when capabilities omitted', async () => {
      // Custom gateway with no capabilities field — must not silently accept images.
      class PlainGw implements IModelGateway {
        called = false
        async complete(): Promise<ModelResponse> {
          this.called = true
          return { content: [{ type: 'text', text: 'x' }], toolCalls: [], finishReason: 'end_turn' }
        }
        async *stream(): AsyncIterable<never> { yield* [] }
      }
      const gw = new PlainGw()
      const port = new DefaultIOPort(gw)
      await expect(port.invokeLLM(visionRequest())).rejects.toMatchObject({
        name: 'ModelGatewayError',
        envelope: {
          code: 'MODEL_CAPABILITY_UNSUPPORTED',
          phase: 'request',
          retryable: false,
          capability: 'imageInput',
        },
      })
      expect(gw.called).toBe(false)

      // Pure text still reaches the custom gateway.
      await expect(port.invokeLLM({
        model: 'm',
        messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      })).resolves.toMatchObject({
        content: [{ type: 'text', text: 'x' }],
      })
      expect(gw.called).toBe(true)
    })

    it('public Milkie.complete routes text+image through adapter and returns ToolCall', async () => {
      const adapter = new OpenAICompatibleAdapter({
        apiKey: 'sk-test',
        capabilities: { imageInput: true },
      })
      const calls: unknown[] = []
      ;(adapter as unknown as { client: { chat: { completions: { create: unknown } } } })
        .client.chat.completions.create = async (params: unknown) => {
          calls.push(params)
          return {
            choices: [{
              message: {
                role: 'assistant',
                content: null,
                tool_calls: [{
                  id: 'call-pub',
                  type: 'function',
                  function: { name: 'look', arguments: '{"ok":true}' },
                }],
              },
              finish_reason: 'tool_calls',
            }],
            usage: { prompt_tokens: 1, completion_tokens: 1 },
          }
        }

      const milkie = new Milkie({
        stateStore: new MemoryStore(),
        gateway: adapter,
      })
      milkie.registerAgent({
        agentId: 'vision-agent',
        version: '1.0.0',
        systemPrompt: 'sys',
        fsm: { states: [{ name: 'react', type: 'llm' }] },
        model: { provider: 'openai-compatible', model: 'vision-model', adapter: 'openai-compatible' },
      })

      const res = await milkie.complete('vision-agent', {
        messages: textImageMessages(),
      })
      expect(res.toolCalls).toEqual([
        expect.objectContaining({ name: 'look', id: 'call-pub' }),
      ])
      const params = calls[0] as { messages: Array<{ role: string; content: unknown }> }
      const user = params.messages.find(m => m.role === 'user')!
      const parts = user.content as Array<{ type: string }>
      expect(parts.map(p => p.type)).toEqual(['text', 'image_url', 'text'])
    })
  })
})
