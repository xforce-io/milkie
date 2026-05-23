import type { ModelConfig } from '../types/agent.js'
import type { IModelGateway } from '../types/model.js'
import { AnthropicAdapter } from './AnthropicAdapter.js'
import { OpenAICompatibleAdapter } from './OpenAICompatibleAdapter.js'

export function createGateway(model: ModelConfig): IModelGateway {
  const adapter = model.adapter.toLowerCase()

  if (adapter === 'anthropic') {
    return new AnthropicAdapter({ baseUrl: model.baseUrl })
  }

  if (
    adapter === 'openai-compatible' ||
    adapter === 'openai' ||
    adapter === 'volcengine'
  ) {
    return new OpenAICompatibleAdapter({
      baseUrl: model.baseUrl ?? process.env['VOLCENGINE_API_BASE'],
      apiKey:
        process.env['VOLCENGINE_TOKEN'] ??
        process.env['OPENAI_API_KEY'],
    })
  }

  throw new Error(`Unknown model adapter: "${model.adapter}"`)
}
