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
      stream:   true,
    })

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta
      if (!delta) continue
      if (delta.content) {
        yield { type: 'message_delta', data: { text: delta.content } }
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

    const usage: ModelUsage | undefined = raw.usage
      ? { inputTokens: raw.usage.prompt_tokens, outputTokens: raw.usage.completion_tokens }
      : undefined

    return { content, toolCalls, usage, finishReason: choice.finish_reason ?? undefined, raw }
  }
}
