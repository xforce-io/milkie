import OpenAI from 'openai'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent, ModelUsage } from '../types/model.js'
import type { MessageContent } from '../types/common.js'
import type { ToolCall } from '../types/tool.js'

export interface OpenAICompatibleAdapterOptions {
  apiKey?:  string
  baseUrl?: string
}

export class OpenAICompatibleAdapter implements IModelGateway {
  private readonly client: OpenAI

  constructor(options: OpenAICompatibleAdapterOptions = {}) {
    this.client = new OpenAI({
      apiKey:  options.apiKey  ?? process.env['VOLCENGINE_TOKEN'] ?? process.env['OPENAI_API_KEY'] ?? '',
      baseURL: options.baseUrl ?? process.env['VOLCENGINE_API_BASE'],
    })
  }

  async complete(request: ModelRequest): Promise<ModelResponse> {
    const raw = await this.client.chat.completions.create({
      model:    request.model,
      messages: this.convertMessages(request),
      tools:    request.tools?.map(t => ({
        type:     'function' as const,
        function: {
          name:        t.name,
          description: t.description,
          parameters:  t.inputSchema,
        },
      })),
      tool_choice: request.tools?.length ? 'auto' : undefined,
    })

    return this.parseResponse(raw)
  }

  async *stream(request: ModelRequest): AsyncIterable<ModelEvent> {
    const stream = await this.client.chat.completions.create({
      model:    request.model,
      messages: this.convertMessages(request),
      tools:    request.tools?.map(t => ({
        type:     'function' as const,
        function: {
          name:        t.name,
          description: t.description,
          parameters:  t.inputSchema,
        },
      })),
      tool_choice:    request.tools?.length ? 'auto' : undefined,
      stream:         true,
      stream_options: { include_usage: true },
    })

    // Accumulate tool_call fragments keyed by their stream `index`.
    const toolCalls = new Map<number, { id: string; argsBuf: string }>()

    for await (const chunk of stream) {
      const choice = chunk.choices[0]
      const delta  = choice?.delta

      if (delta?.content) {
        yield { type: 'message_delta', data: { text: delta.content } }
      }

      for (const tc of delta?.tool_calls ?? []) {
        const index = tc.index
        let entry = toolCalls.get(index)
        if (!entry) {
          // id 仅在该 index 首片读取（OpenAI 协议保证 id 在首片出现，后续片省略）。
          const id = tc.id ?? `idx-${index}`
          entry = { id, argsBuf: '' }
          toolCalls.set(index, entry)
          yield { type: 'tool_call_start', data: { toolCallId: id, name: tc.function?.name ?? '' } }
        }
        const argsPiece = tc.function?.arguments
        if (argsPiece) {
          entry.argsBuf += argsPiece
          yield { type: 'tool_call_delta', data: { toolCallId: entry.id, delta: argsPiece } }
        }
      }

      // finish_reason marks the completion of tool calls — emit done for every
      // accumulated call, then reset for any subsequent independent batch.
      if (choice?.finish_reason && toolCalls.size > 0) {
        for (const entry of toolCalls.values()) {
          let input: unknown = {}
          if (entry.argsBuf) {
            try {
              input = JSON.parse(entry.argsBuf)
            } catch {
              input = {}
            }
          }
          yield { type: 'tool_call_done', data: { toolCallId: entry.id, input } }
        }
        toolCalls.clear()
      }

      const usage = chunk.usage
      if (usage) {
        const cachedTokens = (usage as { prompt_tokens_details?: { cached_tokens?: number } })
          .prompt_tokens_details?.cached_tokens
        yield {
          type: 'usage',
          data: {
            inputTokens:  usage.prompt_tokens,
            outputTokens: usage.completion_tokens,
            ...(cachedTokens !== undefined ? { cacheReadTokens: cachedTokens } : {}),
          },
        }
      }
    }
  }

  private convertMessages(request: ModelRequest): OpenAI.ChatCompletionMessageParam[] {
    const result: OpenAI.ChatCompletionMessageParam[] = []

    if (request.system) {
      result.push({ role: 'system', content: request.system })
    }

    for (const msg of request.messages) {
      if (msg.role === 'tool') {
        for (const c of msg.content) {
          if (c.type === 'tool_result') {
            result.push({
              role:         'tool',
              tool_call_id: c.tool_use_id,
              content:      c.content,
            })
          }
        }
        continue
      }

      if (msg.role === 'assistant') {
        const textParts: string[] = []
        const toolCalls: OpenAI.ChatCompletionMessageToolCall[] = []

        for (const c of msg.content) {
          if (c.type === 'text') {
            textParts.push(c.text)
          } else if (c.type === 'tool_use') {
            toolCalls.push({
              id:       c.id,
              type:     'function',
              function: { name: c.name, arguments: JSON.stringify(c.input) },
            })
          }
        }

        result.push({
          role:       'assistant',
          content:    textParts.join('') || null,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        })
        continue
      }

      // user message
      const text = msg.content
        .filter(c => c.type === 'text')
        .map(c => (c as { type: 'text'; text: string }).text)
        .join('')
      result.push({ role: 'user', content: text })
    }

    return result
  }

  private parseResponse(raw: OpenAI.ChatCompletion): ModelResponse {
    const choice = raw.choices[0]
    if (!choice) throw new Error('OpenAI response has no choices')

    const msg      = choice.message
    const content: MessageContent[] = []
    const toolCalls: ToolCall[] = []

    if (msg.content) {
      content.push({ type: 'text', text: msg.content })
    }

    for (const tc of msg.tool_calls ?? []) {
      if (tc.type !== 'function') continue
      let input: unknown
      try {
        input = JSON.parse(tc.function.arguments)
      } catch {
        input = {}
      }
      content.push({ type: 'tool_use', id: tc.id, name: tc.function.name, input })
      toolCalls.push({ id: tc.id, name: tc.function.name, input })
    }

    // PR-D follow-up: OpenAI's chat completions response includes
    // prompt_tokens_details.cached_tokens when auto prefix caching hits
    // (gpt-4o family, gpt-4o-mini, o1 family). No separate "creation
    // tokens" counter — OpenAI's auto-cache writes are not surfaced.
    const cachedTokens = (raw.usage as { prompt_tokens_details?: { cached_tokens?: number } } | undefined)
      ?.prompt_tokens_details?.cached_tokens
    const usage: ModelUsage | undefined = raw.usage
      ? {
          inputTokens:  raw.usage.prompt_tokens,
          outputTokens: raw.usage.completion_tokens,
          ...(cachedTokens !== undefined ? { cacheReadTokens: cachedTokens } : {}),
        }
      : undefined

    return { content, toolCalls, usage, finishReason: choice.finish_reason ?? undefined, raw }
  }
}
