export const SUFFIX_TO_FIELD = {
  TRANSPORT: 'transport',
  PROTOCOL:  'protocol',
  RUNTIME:   'runtime',
  MODEL:     'model',
  BASE_URL:  'baseUrl',
  API_KEY:   'apiKey',
  PROVIDER:  'provider',
} as const

export type Suffix = keyof typeof SUFFIX_TO_FIELD
export type CanonicalField = typeof SUFFIX_TO_FIELD[Suffix]

export type Transport = 'api' | 'agent-cli'
export type Protocol = 'anthropic-messages' | 'openai-chat-completions'
export type Runtime = 'claude-code' | 'grok-cli' | 'codex'
export type ConnectionSource = 'canonical' | 'legacy'
export type AdapterFamily = 'anthropic' | 'openai-compatible'

export type ConnectionConfigCode =
  | 'CONNECTION_CONFIG_MISSING_FIELD'
  | 'CONNECTION_CONFIG_CONFLICT'
  | 'CONNECTION_CONFIG_UNKNOWN_VALUE'
  | 'CONNECTION_CONFIG_LEGACY_EXPIRED'
  | 'CONNECTION_CONFIG_LEGACY_AND_CANONICAL'

export interface CanonicalFields {
  transport?: string
  protocol?: string
  runtime?: string
  model?: string
  baseUrl?: string
  apiKey?: string
  provider?: string
}

export interface LegacyModelConfig {
  adapter?: string
  model?: string
  provider?: string
  baseUrl?: string
}

export interface ConnectionInput {
  contractVersion?: number
  prefix?: string
  env?: Record<string, string | undefined>
  fields?: Record<string, string | undefined>
  legacyModelConfig?: LegacyModelConfig
  legacyEnv?: Record<string, string | undefined>
}

export interface ConnectionProjection {
  contractVersion: number
  transport: Transport
  protocol?: Protocol
  runtime?: Runtime
  model?: string
  provider?: string
  hasApiKey: boolean
  hasBaseUrl: boolean
  source: ConnectionSource
}

export interface ExecutionMaterials {
  apiKey: string
  baseUrl?: string
}

export const SAFE_CONNECTION_MESSAGES: Record<ConnectionConfigCode, string> = {
  CONNECTION_CONFIG_MISSING_FIELD: 'Model connection configuration is missing a required field.',
  CONNECTION_CONFIG_CONFLICT: 'Model connection configuration has conflicting fields.',
  CONNECTION_CONFIG_UNKNOWN_VALUE: 'Model connection configuration contains an unknown or blank value.',
  CONNECTION_CONFIG_LEGACY_EXPIRED: 'Legacy model connection configuration is outside the migration window.',
  CONNECTION_CONFIG_LEGACY_AND_CANONICAL: 'Canonical and legacy model connection sources were both provided.',
}

export const LEGACY_ENV_KEYS = [
  'ANTHROPIC_API_KEY',
  'OPENAI_API_KEY',
  'VOLCENGINE_TOKEN',
  'VOLCENGINE_API_BASE',
] as const

export type LegacyEnvKey = typeof LEGACY_ENV_KEYS[number]
