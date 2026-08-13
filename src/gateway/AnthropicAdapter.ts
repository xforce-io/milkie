import Anthropic from '@anthropic-ai/sdk'
import type {
  GatewayInvocationOptions,
  IModelGateway,
  ModelCapabilities,
  ModelGatewayCallOptions,
  ModelRequest,
  ModelResponse,
  ModelEvent,
  ModelUsage,
} from '../types/model.js'
import type { Message, MessageContent } from '../types/common.js'
import type { ToolCall } from '../types/tool.js'
import { assertImageRequestSupported } from './imageContent.js'
import { normalizeModelGatewayError } from './ModelGatewayError.js'

export interface AnthropicAdapterOptions {
  apiKey?:  string
  baseUrl?: string
  /**
   * #236: explicit capabilities. imageInput is fail-closed — only `true` enables
   * vision; omitted/undefined/false reject image parts (same as DefaultIOPort).
   */
  capabilities?: ModelCapabilities
}

export class AnthropicAdapter implements IModelGateway {
  private readonly client: Anthropic
  readonly capabilities: ModelCapabilities

  // Cross-event state for streaming tool_use: maps a content block `index` to
  // its tool call id and accumulating partial_json buffer (Anthropic splits
  // tool input across multiple input_json_delta fragments).
  private streamTools: Map<number, { id: string; buf: string }> = new Map()

  constructor(options: AnthropicAdapterOptions = {}) {
    this.capabilities = {
      // Fail closed: require explicit imageInput:true (never infer from provider).
      imageInput: options.capabilities?.imageInput === true,
    }
    this.client = new Anthropic({
      apiKey:  options.apiKey ?? process.env['ANTHROPIC_API_KEY'],
      baseURL: options.baseUrl,
    })
  }

  private guardImageRequest(request: ModelRequest): void {
    assertImageRequestSupported(request, this.capabilities, {
      provider: 'anthropic',
      model: request.model,
    })
  }

  async complete(request: ModelRequest, opts?: ModelGatewayCallOptions | GatewayInvocationOptions): Promise<ModelResponse> {
    this.guardImageRequest(request)
    const params = this.buildParams(request)
    try {
      const raw = await (this.client.messages.create as (p: unknown, o?: { signal?: AbortSignal }) => Promise<Anthropic.Message>)(
        { ...params, stream: false },
        opts?.signal ? { signal: opts.signal } : undefined,
      )
      return this.parseResponse(raw as Anthropic.Message)
    } catch (error) {
      throw normalizeModelGatewayError(error, {
        provider: 'anthropic', model: request.model, phase: 'request',
      })
    }
  }

  async *stream(request: ModelRequest, opts?: ModelGatewayCallOptions | GatewayInvocationOptions): AsyncIterable<ModelEvent> {
    this.guardImageRequest(request)
    const params = this.buildParams(request)
    this.streamTools.clear()
    try {
      const stream = (this.client.messages.stream as (p: unknown, o?: { signal?: AbortSignal }) => AsyncIterable<unknown>)(
        params,
        opts?.signal ? { signal: opts.signal } : undefined,
      )
      for await (const event of stream) {
        yield* this.parseStreamEvent(event)
      }
    } catch (error) {
      throw normalizeModelGatewayError(error, {
        provider: 'anthropic', model: request.model, phase: 'stream_open',
      })
    } finally {
      this.streamTools.clear()
    }
  }

  private buildParams(request: ModelRequest): Record<string, unknown> {
    const params: Record<string, unknown> = {
      model:      request.model,
      max_tokens: 8096,
      messages:   this.convertMessages(request.messages),
    }

    if (request.system) {
      if (request.cacheBreakpoint === 'system-end') {
        params['system'] = [{
          type:          'text',
          text:          request.system,
          cache_control: { type: 'ephemeral' },
        }]
      } else {
        params['system'] = request.system
      }
    }

    if (request.tools && request.tools.length > 0) {
      params['tools'] = request.tools.map(t => ({
        name:         t.name,
        description:  t.description,
        input_schema: t.inputSchema,
      }))
    }

    if (request.toolChoice) {
      params['tool_choice'] = request.toolChoice
    }

    // #126: forward sampling temperature when set; omit otherwise (provider default).
    if (request.temperature !== undefined) {
      params['temperature'] = request.temperature
    }

    return params
  }

  private convertMessages(messages: Message[]): unknown[] {
    const result: unknown[] = []

    for (const msg of messages) {
      if (msg.role === 'tool') {
        const content = msg.content.map(c => {
          if (c.type === 'tool_result') {
            return {
              type:        'tool_result',
              tool_use_id: c.tool_use_id,
              content:     c.content,
              is_error:    c.is_error ?? false,
            }
          }
          return c
        })
        result.push({ role: 'user', content })
        continue
      }

      const content = msg.content.map(c => {
        if (c.type === 'tool_use') {
          return { type: 'tool_use', id: c.id, name: c.name, input: c.input }
        }
        if (c.type === 'text') {
          return { type: 'text', text: c.text }
        }
        if (c.type === 'image') {
          if (c.source.kind === 'url') {
            return {
              type: 'image',
              source: { type: 'url', url: c.source.url },
            }
          }
          return {
            type: 'image',
            source: {
              type: 'base64',
              media_type: c.mediaType,
              data: c.source.data,
            },
          }
        }
        return c
      })

      result.push({ role: msg.role, content })
    }

    return result
  }

  private parseResponse(raw: Anthropic.Message): ModelResponse {
    const content: MessageContent[] = []
    const toolCalls: ToolCall[] = []

    for (const block of raw.content) {
      if (block.type === 'text') {
        content.push({ type: 'text', text: block.text })
      } else if (block.type === 'tool_use') {
        content.push({ type: 'tool_use', id: block.id, name: block.name, input: block.input })
        toolCalls.push({ id: block.id, name: block.name, input: block.input })
      }
    }

    // Anthropic reports cache_read_input_tokens / cache_creation_input_tokens
    // alongside input_tokens whenever the request used cache_control markers
    // (either via milkie's cacheBreakpoint plumbing or set externally). The
    // raw API types in older @anthropic-ai/sdk versions don't always declare
    // these fields, so read defensively through an unknown cast.
    const rawUsage = raw.usage as {
      input_tokens:                  number
      output_tokens:                 number
      cache_read_input_tokens?:      number | null
      cache_creation_input_tokens?:  number | null
    }
    const cacheReadTokens     = rawUsage.cache_read_input_tokens     ?? undefined
    const cacheCreationTokens = rawUsage.cache_creation_input_tokens ?? undefined

    const usage: ModelUsage = {
      inputTokens:  rawUsage.input_tokens,
      outputTokens: rawUsage.output_tokens,
      ...(cacheReadTokens     !== undefined ? { cacheReadTokens }     : {}),
      ...(cacheCreationTokens !== undefined ? { cacheCreationTokens } : {}),
    }

    return { content, toolCalls, usage, finishReason: raw.stop_reason ?? undefined, raw }
  }

  private *parseStreamEvent(event: unknown): Iterable<ModelEvent> {
    const e = event as { type: string; index?: number; [k: string]: unknown }

    if (e.type === 'content_block_start') {
      const block = e['content_block'] as { type?: string; id?: string; name?: string } | undefined
      if (block?.type === 'tool_use' && e.index !== undefined) {
        const id = block.id ?? ''
        this.streamTools.set(e.index, { id, buf: '' })
        yield { type: 'tool_call_start', data: { toolCallId: id, name: block.name ?? '' } }
      }
    } else if (e.type === 'content_block_delta') {
      const delta = e['delta'] as { type: string; text?: string; partial_json?: string } | undefined
      if (delta?.type === 'text_delta' && delta.text) {
        yield { type: 'message_delta', data: { text: delta.text } }
      } else if (delta?.type === 'input_json_delta' && delta.partial_json !== undefined && e.index !== undefined) {
        const slot = this.streamTools.get(e.index)
        if (slot) {
          slot.buf += delta.partial_json
          yield { type: 'tool_call_delta', data: { toolCallId: slot.id, delta: delta.partial_json } }
        }
      }
    } else if (e.type === 'content_block_stop') {
      if (e.index !== undefined) {
        const slot = this.streamTools.get(e.index)
        if (slot) {
          let input: unknown = {}
          let invalidArguments: ToolCall['invalidArguments']
          if (slot.buf === '') {
            invalidArguments = {
              code:      'TOOL_ARGUMENTS_INVALID_JSON',
              message:   'Tool arguments are not valid JSON',
              rawLength: 0,
            }
          } else {
            try {
              input = JSON.parse(slot.buf)
            } catch {
              invalidArguments = {
                code:      'TOOL_ARGUMENTS_INVALID_JSON',
                message:   'Tool arguments are not valid JSON',
                rawLength: slot.buf.length,
              }
            }
          }
          this.streamTools.delete(e.index)
          yield {
            type: 'tool_call_done',
            data: {
              toolCallId: slot.id,
              input,
              ...(invalidArguments !== undefined ? { invalidArguments } : {}),
            },
          }
        }
      }
    } else if (e.type === 'message_delta') {
      const usage = (e['usage'] as { output_tokens?: number } | undefined)
      if (usage) {
        yield { type: 'usage', data: { inputTokens: 0, outputTokens: usage.output_tokens ?? 0 } }
      }
    }
  }
}
