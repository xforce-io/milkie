import { ConnectionConfigError } from './errors.js'
import {
  LEGACY_ENV_KEYS,
  SUFFIX_TO_FIELD,
  type AdapterFamily,
  type CanonicalField,
  type CanonicalFields,
  type ConnectionInput,
  type ConnectionProjection,
  type LegacyEnvKey,
  type Protocol,
  type Runtime,
  type Suffix,
  type Transport,
} from './types.js'

const MATERIALS = new WeakMap<ConnectionParseResult, { apiKey: string; baseUrl?: string }>()

export function takeExecutionMaterials(parsed: ConnectionParseResult): { apiKey: string; baseUrl?: string } | undefined {
  const materials = MATERIALS.get(parsed)
  MATERIALS.delete(parsed)
  return materials
}

const FIELD_BY_SUFFIX = SUFFIX_TO_FIELD
const SUFFIXES = Object.keys(FIELD_BY_SUFFIX) as Suffix[]
const CANONICAL_FIELDS = Object.values(FIELD_BY_SUFFIX) as CanonicalField[]
const CANONICAL_FIELD_SET: Record<string, true> = {
  transport: true,
  protocol: true,
  runtime: true,
  model: true,
  baseUrl: true,
  apiKey: true,
  provider: true,
}
const LEGACY_ENV_SET: Record<string, true> = {
  ANTHROPIC_API_KEY: true,
  OPENAI_API_KEY: true,
  VOLCENGINE_TOKEN: true,
  VOLCENGINE_API_BASE: true,
}
const TRANSPORTS: Record<string, true> = { api: true, 'agent-cli': true }
const PROTOCOLS: Record<string, AdapterFamily> = {
  'anthropic-messages': 'anthropic',
  'openai-chat-completions': 'openai-compatible',
}
const RUNTIMES: Record<string, true> = { 'claude-code': true, 'grok-cli': true, codex: true }
const ADAPTER_TO_PROTOCOL: Record<string, Protocol> = {
  anthropic: 'anthropic-messages',
  'openai-compatible': 'openai-chat-completions',
  openai: 'openai-chat-completions',
  volcengine: 'openai-chat-completions',
}

export interface ConnectionParseResult {
  projection: ConnectionProjection
}

function fail(code: ConnectionConfigError['code'], fields: string[]): never {
  throw new ConnectionConfigError(code, fields)
}

function isWholeNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value)
}

function isBlank(value: string): boolean {
  return value.length === 0 || value !== value.trim()
}

function presentKeys(bag: Record<string, string | undefined> | undefined): string[] {
  if (!bag) return []
  return Object.keys(bag).filter(key => bag[key] !== undefined)
}

function copyDefined(bag: Record<string, string | undefined> | undefined): Record<string, string> {
  const out: Record<string, string> = {}
  if (!bag) return out
  for (const key of Object.keys(bag)) {
    const value = bag[key]
    if (value !== undefined) out[key] = value
  }
  return out
}

export function collectFromPrefix(
  prefix: string,
  env: Record<string, string | undefined>,
): CanonicalFields {
  const fields: CanonicalFields = {}
  for (const suffix of SUFFIXES) {
    const raw = env[prefix + suffix]
    if (raw === undefined) continue
    fields[FIELD_BY_SUFFIX[suffix]] = raw
  }
  return fields
}

function collectLegacyEnv(
  bag: Record<string, string | undefined> | undefined,
): Partial<Record<LegacyEnvKey, string>> {
  const out: Partial<Record<LegacyEnvKey, string>> = {}
  if (!bag) return out
  for (const key of LEGACY_ENV_KEYS) {
    const value = bag[key]
    if (value !== undefined) out[key] = value
  }
  return out
}

function setMapped(
  target: CanonicalFields,
  field: CanonicalField,
  value: string,
  origin: string,
  origins: Partial<Record<CanonicalField, string[]>>,
): void {
  const current = target[field]
  if (current !== undefined && current !== value) {
    fail('CONNECTION_CONFIG_CONFLICT', [field, ...origins[field] ?? [], origin])
  }
  target[field] = value
  origins[field] = [...(origins[field] ?? []), origin]
}

function materializeLegacy(
  model: ConnectionInput['legacyModelConfig'],
  legacyEnv: Partial<Record<LegacyEnvKey, string>>,
): CanonicalFields {
  const fields: CanonicalFields = {}
  const origins: Partial<Record<CanonicalField, string[]>> = {}
  if (model?.adapter !== undefined) {
    const protocol = ADAPTER_TO_PROTOCOL[model.adapter]
    if (!protocol) fail('CONNECTION_CONFIG_UNKNOWN_VALUE', ['legacyModelConfig'])
    setMapped(fields, 'transport', 'api', 'legacyModelConfig', origins)
    setMapped(fields, 'protocol', protocol, 'legacyModelConfig', origins)
  }
  if (model?.model !== undefined) setMapped(fields, 'model', model.model, 'legacyModelConfig', origins)
  if (model?.provider !== undefined) setMapped(fields, 'provider', model.provider, 'legacyModelConfig', origins)
  if (model?.baseUrl !== undefined) setMapped(fields, 'baseUrl', model.baseUrl, 'legacyModelConfig', origins)
  if (legacyEnv.ANTHROPIC_API_KEY !== undefined) {
    setMapped(fields, 'apiKey', legacyEnv.ANTHROPIC_API_KEY, 'ANTHROPIC_API_KEY', origins)
  }
  if (legacyEnv.OPENAI_API_KEY !== undefined) {
    setMapped(fields, 'apiKey', legacyEnv.OPENAI_API_KEY, 'OPENAI_API_KEY', origins)
  }
  if (legacyEnv.VOLCENGINE_TOKEN !== undefined) {
    setMapped(fields, 'apiKey', legacyEnv.VOLCENGINE_TOKEN, 'VOLCENGINE_TOKEN', origins)
  }
  if (legacyEnv.VOLCENGINE_API_BASE !== undefined) {
    setMapped(fields, 'baseUrl', legacyEnv.VOLCENGINE_API_BASE, 'VOLCENGINE_API_BASE', origins)
  }
  return fields
}

function parseFields(contractVersion: number, fields: CanonicalFields, source: 'canonical' | 'legacy'): ConnectionParseResult {
  const blank: string[] = []
  for (const name of CANONICAL_FIELDS) {
    const value = fields[name]
    if (value !== undefined && isBlank(value)) blank.push(name)
  }
  if (blank.length > 0) fail('CONNECTION_CONFIG_UNKNOWN_VALUE', blank)

  const unknownEnum: string[] = []
  if (fields.transport !== undefined && !TRANSPORTS[fields.transport]) unknownEnum.push('transport')
  if (fields.protocol !== undefined && !PROTOCOLS[fields.protocol]) unknownEnum.push('protocol')
  if (fields.runtime !== undefined && !RUNTIMES[fields.runtime]) unknownEnum.push('runtime')
  if (unknownEnum.length > 0) fail('CONNECTION_CONFIG_UNKNOWN_VALUE', unknownEnum)

  const conflicts: string[] = []
  if (fields.transport === 'api' && fields.runtime !== undefined) conflicts.push('runtime')
  if (fields.transport === 'agent-cli') {
    if (fields.protocol !== undefined) conflicts.push('protocol')
    if (fields.apiKey !== undefined) conflicts.push('apiKey')
    if (fields.baseUrl !== undefined) conflicts.push('baseUrl')
  }
  if (conflicts.length > 0) fail('CONNECTION_CONFIG_CONFLICT', conflicts)

  const missing: string[] = []
  if (fields.transport === undefined) missing.push('transport')
  if (fields.transport === 'api') {
    if (fields.protocol === undefined) missing.push('protocol')
    if (fields.model === undefined) missing.push('model')
    if (fields.apiKey === undefined) missing.push('apiKey')
  }
  if (fields.transport === 'agent-cli' && fields.runtime === undefined) missing.push('runtime')
  if (missing.length > 0) fail('CONNECTION_CONFIG_MISSING_FIELD', missing)

  const transport = fields.transport as Transport
  const projection: ConnectionProjection = {
    contractVersion,
    transport,
    hasApiKey: fields.apiKey !== undefined,
    hasBaseUrl: fields.baseUrl !== undefined,
    source,
  }
  if (fields.protocol !== undefined) projection.protocol = fields.protocol as Protocol
  if (fields.runtime !== undefined) projection.runtime = fields.runtime as Runtime
  if (fields.model !== undefined) projection.model = fields.model
  if (fields.provider !== undefined) projection.provider = fields.provider

  const result: ConnectionParseResult = { projection }
  if (transport === 'api' && fields.apiKey !== undefined) {
    MATERIALS.set(result, {
      apiKey: fields.apiKey,
      ...(fields.baseUrl !== undefined ? { baseUrl: fields.baseUrl } : {}),
    })
  }
  return result
}

export function resolveAndParseConnection(input: ConnectionInput): ConnectionParseResult {
  if (input.contractVersion === undefined) fail('CONNECTION_CONFIG_MISSING_FIELD', ['contractVersion'])
  if (!isWholeNumber(input.contractVersion) || input.contractVersion < 1) {
    fail('CONNECTION_CONFIG_UNKNOWN_VALUE', ['contractVersion'])
  }

  const hasPrefix = input.prefix !== undefined
  const hasEnv = input.env !== undefined
  const hasFields = input.fields !== undefined
  const hasLegacyEnv = input.legacyEnv !== undefined
  const entryA = hasPrefix || hasEnv
  const entryB = hasFields
  const shape: string[] = []
  if (entryA === entryB) shape.push('entry')
  if (hasPrefix && !hasEnv) shape.push('env')
  if (hasEnv && !hasPrefix) shape.push('prefix')
  if (entryA && hasLegacyEnv) shape.push('legacyEnv')
  if (entryB && hasPrefix) shape.push('prefix')
  if (entryB && hasEnv) shape.push('env')
  if (shape.length > 0) fail('CONNECTION_CONFIG_CONFLICT', shape)

  let canonical: CanonicalFields
  let legacyBag: Record<string, string>
  if (entryA) {
    canonical = collectFromPrefix(input.prefix as string, input.env as Record<string, string | undefined>)
    legacyBag = copyDefined(collectLegacyEnv(input.env))
  } else {
    const extra = presentKeys(input.fields).filter(key => !CANONICAL_FIELD_SET[key])
    if (extra.length > 0) fail('CONNECTION_CONFIG_UNKNOWN_VALUE', extra)
    const unknownLegacy = presentKeys(input.legacyEnv).filter(key => !LEGACY_ENV_SET[key])
    if (unknownLegacy.length > 0) fail('CONNECTION_CONFIG_UNKNOWN_VALUE', unknownLegacy)
    canonical = {}
    for (const name of CANONICAL_FIELDS) {
      const value = input.fields?.[name]
      if (value !== undefined) canonical[name] = value
    }
    legacyBag = copyDefined(collectLegacyEnv(input.legacyEnv))
  }

  const canonicalNames = CANONICAL_FIELDS.filter(name => canonical[name] !== undefined)
  const legacyNames = [
    ...Object.keys(legacyBag),
    ...(input.legacyModelConfig !== undefined ? ['legacyModelConfig'] : []),
  ]
  const hasCanonical = canonicalNames.length > 0
  const hasLegacy = legacyNames.length > 0

  if (hasCanonical && hasLegacy) fail('CONNECTION_CONFIG_LEGACY_AND_CANONICAL', [...canonicalNames, ...legacyNames])
  if (hasLegacy && input.contractVersion >= 2) fail('CONNECTION_CONFIG_LEGACY_EXPIRED', legacyNames)

  if (hasLegacy) {
    const blanks: string[] = []
    if (input.legacyModelConfig) {
      for (const key of ['adapter', 'model', 'provider', 'baseUrl'] as const) {
        const value = input.legacyModelConfig[key]
        if (value !== undefined && isBlank(value)) blanks.push('legacyModelConfig')
      }
    }
    for (const key of Object.keys(legacyBag)) {
      if (isBlank(legacyBag[key] as string)) blanks.push(key)
    }
    if (blanks.length > 0) fail('CONNECTION_CONFIG_UNKNOWN_VALUE', blanks)
    return parseFields(input.contractVersion, materializeLegacy(input.legacyModelConfig, legacyBag), 'legacy')
  }

  return parseFields(input.contractVersion, canonical, 'canonical')
}
