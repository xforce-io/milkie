import OpenAI from 'openai'
import type {
  IModelGateway,
  ModelCapabilities,
  ModelGatewayCallOptions,
  ModelRequest,
  ModelResponse,
  ModelEvent,
  ModelUsage,
} from '../types/model.js'
import type { MessageContent } from '../types/common.js'
import type { ToolCall } from '../types/tool.js'
import { normalizeModelGatewayError } from './ModelGatewayError.js'
import { assertImageRequestSupported } from './imageContent.js'

export interface OpenAICompatibleAdapterOptions {
  apiKey?:  string
  baseUrl?: string
  provider?: string
  /**
   * #236: explicit capabilities. Never inferred from endpoint/provider.
   * Default imageInput:false — set true only when the deployed model accepts vision.
   */
  capabilities?: ModelCapabilities
}

export class OpenAICompatibleAdapter implements IModelGateway {
  private readonly client: OpenAI
  private readonly provider: string
  readonly capabilities: ModelCapabilities
  /** Default completion budget — long tool-arg payloads (e.g. JSON writes) need headroom above common 4k gateway defaults. */
  static readonly DEFAULT_MAX_TOKENS = 8192

  constructor(options: OpenAICompatibleAdapterOptions = {}) {
    this.provider = options.provider ?? 'openai-compatible'
    this.capabilities = {
      imageInput: options.capabilities?.imageInput === true,
    }
    this.client = new OpenAI({
      apiKey:  options.apiKey  ?? process.env['VOLCENGINE_TOKEN'] ?? process.env['OPENAI_API_KEY'] ?? '',
      baseURL: options.baseUrl ?? process.env['VOLCENGINE_API_BASE'],
    })
  }

  private guardImageRequest(request: ModelRequest): void {
    assertImageRequestSupported(request, this.capabilities, {
      provider: this.provider,
      model: request.model,
    })
  }

  async complete(request: ModelRequest, opts?: ModelGatewayCallOptions): Promise<ModelResponse> {
    this.guardImageRequest(request)
    let raw: OpenAI.ChatCompletion
    try {
      raw = await this.client.chat.completions.create({
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
        temperature: request.temperature,
        max_tokens:  request.maxTokens ?? OpenAICompatibleAdapter.DEFAULT_MAX_TOKENS,
      }, opts?.signal ? { signal: opts.signal } : undefined)
    } catch (error) {
      throw normalizeModelGatewayError(error, {
        provider: this.provider, model: request.model, phase: 'request',
      })
    }

    try {
      return this.parseResponse(raw)
    } catch (error) {
      throw normalizeModelGatewayError(error, {
        provider: this.provider, model: request.model, phase: 'response_parse',
      })
    }
  }

  async *stream(request: ModelRequest, opts?: ModelGatewayCallOptions): AsyncIterable<ModelEvent> {
    this.guardImageRequest(request)
    let stream: AsyncIterable<OpenAI.ChatCompletionChunk>
    try {
      stream = await this.client.chat.completions.create({
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
        temperature:    request.temperature,
        max_tokens:     request.maxTokens ?? OpenAICompatibleAdapter.DEFAULT_MAX_TOKENS,
        stream:         true,
        stream_options: { include_usage: true },
      }, opts?.signal ? { signal: opts.signal } : undefined)
    } catch (error) {
      throw normalizeModelGatewayError(error, {
        provider: this.provider, model: request.model, phase: 'stream_open',
      })
    }

    // Accumulate tool_call fragments keyed by their stream `index`.
    const toolCalls = new Map<number, { id: string; argsBuf: string; sawArguments: boolean }>()

    try {
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
          entry = { id, argsBuf: '', sawArguments: false }
          toolCalls.set(index, entry)
          yield { type: 'tool_call_start', data: { toolCallId: id, name: tc.function?.name ?? '' } }
        }
        const argsPiece = tc.function?.arguments
        if (argsPiece !== undefined) {
          entry.sawArguments = true
          if (argsPiece) {
            entry.argsBuf += argsPiece
            yield { type: 'tool_call_delta', data: { toolCallId: entry.id, delta: argsPiece } }
          }
        }
      }

      // finish_reason marks the completion of tool calls — emit done for every
      // accumulated call, then reset for any subsequent independent batch.
      if (choice?.finish_reason && toolCalls.size > 0) {
        const finishReason = choice.finish_reason
        for (const entry of toolCalls.values()) {
          let input: unknown = {}
          let invalidArguments: ToolCall['invalidArguments']
          if (!entry.sawArguments || entry.argsBuf === '') {
            invalidArguments = {
              code:      finishReason === 'length' ? 'TOOL_ARGUMENTS_TRUNCATED' : 'TOOL_ARGUMENTS_INVALID_JSON',
              message:   finishReason === 'length'
                ? 'Tool arguments were truncated before a complete JSON object was produced'
                : 'Tool arguments are not valid JSON',
              rawLength: 0,
            }
          } else {
            try {
              input = JSON.parse(entry.argsBuf)
            } catch {
              invalidArguments = {
                code:      finishReason === 'length' ? 'TOOL_ARGUMENTS_TRUNCATED' : 'TOOL_ARGUMENTS_INVALID_JSON',
                message:   finishReason === 'length'
                  ? 'Tool arguments were truncated before a complete JSON object was produced'
                  : 'Tool arguments are not valid JSON',
                rawLength: entry.argsBuf.length,
              }
            }
          }
          yield {
            type: 'tool_call_done',
            data: {
              toolCallId: entry.id,
              input,
              ...(invalidArguments !== undefined ? { invalidArguments } : {}),
            },
          }
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
    } catch (error) {
      throw normalizeModelGatewayError(error, {
        provider: this.provider, model: request.model, phase: 'stream_read',
      })
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
        // #236: preserve text/image order for assistant multimodal history.
        // Never silently drop image parts. tool_use still maps to tool_calls.
        const contentParts: OpenAI.ChatCompletionContentPart[] = []
        const toolCalls: OpenAI.ChatCompletionMessageToolCall[] = []
        let hasImage = false

        for (const c of msg.content) {
          if (c.type === 'text') {
            contentParts.push({ type: 'text', text: c.text })
          } else if (c.type === 'image') {
            hasImage = true
            const url = c.source.kind === 'url'
              ? c.source.url
              : `data:${c.mediaType};base64,${c.source.data}`
            contentParts.push({ type: 'image_url', image_url: { url } })
          } else if (c.type === 'tool_use') {
            toolCalls.push({
              id:       c.id,
              type:     'function',
              function: { name: c.name, arguments: JSON.stringify(c.input) },
            })
          }
        }

        if (hasImage) {
          result.push({
            role:       'assistant',
            content:    contentParts,
            tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
          } as OpenAI.ChatCompletionMessageParam)
        } else {
          const textJoined = contentParts
            .filter((p): p is OpenAI.ChatCompletionContentPartText => p.type === 'text')
            .map(p => p.text)
            .join('')
          result.push({
            role:       'assistant',
            content:    textJoined || null,
            tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
          })
        }
        continue
      }

      // user message — preserve text/image order for vision models (#236)
      const parts: OpenAI.ChatCompletionContentPart[] = []
      for (const c of msg.content) {
        if (c.type === 'text') {
          parts.push({ type: 'text', text: c.text })
        } else if (c.type === 'image') {
          const url = c.source.kind === 'url'
            ? c.source.url
            : `data:${c.mediaType};base64,${c.source.data}`
          parts.push({ type: 'image_url', image_url: { url } })
        }
      }
      if (parts.length === 1 && parts[0]!.type === 'text') {
        result.push({ role: 'user', content: parts[0]!.text })
      } else {
        result.push({ role: 'user', content: parts })
      }
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
      let input: unknown = {}
      let invalidArguments: ToolCall['invalidArguments']
      const argumentsText = tc.function.arguments
      const finishReason = choice.finish_reason ?? undefined
      if (typeof argumentsText !== 'string') {
        invalidArguments = {
          code:      finishReason === 'length' ? 'TOOL_ARGUMENTS_TRUNCATED' : 'TOOL_ARGUMENTS_INVALID_JSON',
          message:   finishReason === 'length'
            ? 'Tool arguments were truncated before a complete JSON object was produced'
            : 'Tool arguments are not valid JSON',
          rawLength: 0,
        }
      } else if (argumentsText === '') {
        invalidArguments = {
          code:      finishReason === 'length' ? 'TOOL_ARGUMENTS_TRUNCATED' : 'TOOL_ARGUMENTS_INVALID_JSON',
          message:   finishReason === 'length'
            ? 'Tool arguments were truncated before a complete JSON object was produced'
            : 'Tool arguments are not valid JSON',
          rawLength: 0,
        }
      } else {
        try {
          input = JSON.parse(argumentsText)
        } catch {
          invalidArguments = {
            code:      finishReason === 'length' ? 'TOOL_ARGUMENTS_TRUNCATED' : 'TOOL_ARGUMENTS_INVALID_JSON',
            message:   finishReason === 'length'
              ? 'Tool arguments were truncated before a complete JSON object was produced'
              : 'Tool arguments are not valid JSON',
            rawLength: argumentsText.length,
          }
        }
      }
      content.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input,
        ...(invalidArguments !== undefined ? { invalidArguments } : {}),
      })
      toolCalls.push({
        id: tc.id,
        name: tc.function.name,
        input,
        ...(invalidArguments !== undefined ? { invalidArguments } : {}),
      })
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

    return { content, toolCalls, usage, finishReason: choice.finish_reason ?? undefined }
  }
}
