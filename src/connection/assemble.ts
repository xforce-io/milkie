import { AnthropicAdapter } from '../gateway/AnthropicAdapter.js'
import { OpenAICompatibleAdapter } from '../gateway/OpenAICompatibleAdapter.js'
import { LoggingGateway } from '../logging/LoggingGateway.js'
import { getLogger } from '../logging/logger.js'
import type { IModelGateway } from '../types/model.js'
import { ConnectionConfigError } from './errors.js'
import { takeExecutionMaterials, type ConnectionParseResult } from './parse.js'
import type { AdapterFamily } from './types.js'

export interface AssembledApiGateway {
  adapterFamily: AdapterFamily
  gateway: IModelGateway
}

export function assembleApiGateway(parsed: ConnectionParseResult): AssembledApiGateway {
  const materials = takeExecutionMaterials(parsed)
  if (parsed.projection.transport !== 'api' || !parsed.projection.protocol || !materials) {
    throw new ConnectionConfigError('CONNECTION_CONFIG_MISSING_FIELD', ['protocol'])
  }
  const protocol = parsed.projection.protocol
  const inner = protocol === 'anthropic-messages'
    ? new AnthropicAdapter({ ...materials, readEnv: false })
    : new OpenAICompatibleAdapter({ ...materials, readEnv: false })
  return {
    adapterFamily: protocol === 'anthropic-messages' ? 'anthropic' : 'openai-compatible',
    gateway: new LoggingGateway(inner, getLogger().child({ mod: 'gateway' })),
  }
}
