import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../types/model.js'
import type { ServiceLogger } from './logger.js'

/**
 * #79：在 gateway 统一包装点打 LLM wide event（每调用一条：model/durationMs/token），
 * 两个 adapter 一处覆盖。只记元数据，不携带 prompt/completion 正文（脱敏，设计 §6）。
 */
export class LoggingGateway implements IModelGateway {
  constructor(
    private readonly inner: IModelGateway,
    private readonly log:   ServiceLogger,
  ) {}

  async complete(request: ModelRequest): Promise<ModelResponse> {
    const startedAt = Date.now()
    try {
      const res = await this.inner.complete(request)
      this.log.info({
        model: request.model, durationMs: Date.now() - startedAt,
        inputTokens: res.usage?.inputTokens, outputTokens: res.usage?.outputTokens,
      }, 'llm call')
      return res
    } catch (err) {
      this.log.error({ model: request.model, durationMs: Date.now() - startedAt, err }, 'llm call failed')
      throw err
    }
  }

  async *stream(request: ModelRequest): AsyncIterable<ModelEvent> {
    const startedAt = Date.now()
    let inputTokens = 0
    let outputTokens = 0
    try {
      for await (const e of this.inner.stream(request)) {
        if (e.type === 'usage') {
          inputTokens  += e.data.inputTokens
          outputTokens += e.data.outputTokens
        }
        yield e
      }
      this.log.info({ model: request.model, durationMs: Date.now() - startedAt, inputTokens, outputTokens }, 'llm stream')
    } catch (err) {
      this.log.error({ model: request.model, durationMs: Date.now() - startedAt, err }, 'llm stream failed')
      throw err
    }
  }
}
