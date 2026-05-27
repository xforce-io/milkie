import Anthropic from '@anthropic-ai/sdk'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent, ModelUsage } from '../types/model.js'
import type { Message, MessageContent } from '../types/common.js'
import type { ToolCall } from '../types/tool.js'

export interface AnthropicAdapterOptions {
  apiKey?:  string
  baseUrl?: string
}

export class AnthropicAdapter implements IModelGateway {
  private readonly client: Anthropic

  constructor(options: AnthropicAdapterOptions = {}) {
    this.client = new Anthropic({
      apiKey:  options.apiKey ?? process.env['ANTHROPIC_API_KEY'],
      baseURL: options.baseUrl,
    })
  }

  async complete(request: ModelRequest): Promise<ModelResponse> {
    const params = this.buildParams(request)
    const raw = await (this.client.messages.create as (p: unknown) => Promise<Anthropic.Message>)({
      ...params,
      stream: false,
    })

    return this.parseResponse(raw as Anthropic.Message)
  }

  async *stream(request: ModelRequest): AsyncIterable<ModelEvent> {
    const params = this.buildParams(request)
    const stream = (this.client.messages.stream as (p: unknown) => AsyncIterable<unknown>)(params)

    for await (const event of stream) {
      yield* this.parseStreamEvent(event)
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
    const e = event as { type: string; [k: string]: unknown }
    if (e.type === 'content_block_delta') {
      const delta = e['delta'] as { type: string; text?: string } | undefined
      if (delta?.type === 'text_delta' && delta.text) {
        yield { type: 'message_delta', data: { text: delta.text } }
      }
    } else if (e.type === 'message_delta') {
      const usage = (e['usage'] as { output_tokens?: number } | undefined)
      if (usage) {
        yield { type: 'usage', data: { inputTokens: 0, outputTokens: usage.output_tokens ?? 0 } }
      }
    }
  }
}
