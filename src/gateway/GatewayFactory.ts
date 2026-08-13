import type { ModelConfig } from '../types/agent.js'
import type { IModelGateway, ModelCapabilities } from '../types/model.js'
import { AnthropicAdapter } from './AnthropicAdapter.js'
import { OpenAICompatibleAdapter } from './OpenAICompatibleAdapter.js'
import { LoggingGateway } from '../logging/LoggingGateway.js'
import { getLogger, type ServiceLogger } from '../logging/logger.js'

function capabilitiesFromModel(model: ModelConfig): ModelCapabilities | undefined {
  if (!model.capabilities) return undefined
  return {
    imageInput: model.capabilities.imageInput === true,
  }
}

export function createGateway(model: ModelConfig, logger?: ServiceLogger): IModelGateway {
  const adapter = model.adapter.toLowerCase()
  // #79：统一在工厂出口包 LoggingGateway，两个 adapter 一处覆盖。
  // 注入的 gateway（MilkieOptions.gateway，测试用）不经过这里，因此不被包装。
  const log = logger ?? getLogger().child({ mod: 'gateway' })
  const capabilities = capabilitiesFromModel(model)

  if (adapter === 'anthropic') {
    return new LoggingGateway(new AnthropicAdapter({
      baseUrl: model.baseUrl,
      ...(capabilities ? { capabilities } : {}),
    }), log)
  }

  if (
    adapter === 'openai-compatible' ||
    adapter === 'openai' ||
    adapter === 'volcengine'
  ) {
    return new LoggingGateway(new OpenAICompatibleAdapter({
      provider: model.provider,
      baseUrl: model.baseUrl ?? process.env['VOLCENGINE_API_BASE'],
      apiKey:
        process.env['VOLCENGINE_TOKEN'] ??
        process.env['OPENAI_API_KEY'],
      ...(capabilities ? { capabilities } : {}),
    }), log)
  }

  throw new Error(`Unknown model adapter: "${model.adapter}"`)
}
