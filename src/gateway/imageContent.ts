import type {
  ImageMediaType,
  ImageSource,
  Message,
  MessageContent,
} from '../types/common.js'
import type { ModelCapabilities, ModelRequest } from '../types/model.js'
import { ModelGatewayError } from './ModelGatewayError.js'

export const SUPPORTED_IMAGE_MEDIA_TYPES: readonly ImageMediaType[] = [
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/gif',
]

const MEDIA_TYPE_SET = new Set<string>(SUPPORTED_IMAGE_MEDIA_TYPES)

export function isImageMediaType(value: string): value is ImageMediaType {
  return MEDIA_TYPE_SET.has(value)
}

export function messageHasImageParts(messages: readonly Message[]): boolean {
  for (const msg of messages) {
    for (const part of msg.content) {
      if (part.type === 'image') return true
    }
  }
  return false
}

export function resolveImageInputCapability(
  capabilities: ModelCapabilities | undefined,
): boolean {
  return capabilities?.imageInput === true
}

/**
 * Validate image parts before any network call.
 * Capability failures use MODEL_CAPABILITY_UNSUPPORTED; format failures use MODEL_BAD_RESPONSE.
 */
export function assertImageRequestSupported(
  request: ModelRequest,
  capabilities: ModelCapabilities | undefined,
  context: { provider: string; model: string },
): void {
  if (!messageHasImageParts(request.messages)) return

  if (!resolveImageInputCapability(capabilities)) {
    throw new ModelGatewayError({
      code:       'MODEL_CAPABILITY_UNSUPPORTED',
      message:    'Model gateway does not support image input.',
      phase:      'request',
      provider:   context.provider,
      model:      context.model,
      retryable:  false,
      capability: 'imageInput',
    })
  }

  for (const msg of request.messages) {
    for (const part of msg.content) {
      if (part.type === 'image') validateImagePart(part, context)
    }
  }
}

function validateImagePart(
  part: Extract<MessageContent, { type: 'image' }>,
  context: { provider: string; model: string },
): void {
  if (!isImageMediaType(part.mediaType)) {
    throw formatError(
      context,
      `Unsupported image mediaType "${String(part.mediaType)}". Supported: ${SUPPORTED_IMAGE_MEDIA_TYPES.join(', ')}`,
    )
  }
  validateImageSource(part.source, context)
}

function validateImageSource(
  source: ImageSource,
  context: { provider: string; model: string },
): void {
  if (!source || typeof source !== 'object') {
    throw formatError(context, 'Image source is required')
  }
  if (source.kind === 'url') {
    if (typeof source.url !== 'string' || source.url.length === 0) {
      throw formatError(context, 'Image URL source requires a non-empty url string')
    }
    let parsed: URL
    try {
      parsed = new URL(source.url)
    } catch {
      throw formatError(context, 'Image URL is not a valid absolute URL')
    }
    if (parsed.protocol !== 'https:') {
      throw formatError(context, 'Image URL must use https://')
    }
    return
  }
  if (source.kind === 'base64') {
    if (typeof source.data !== 'string' || source.data.length === 0) {
      throw formatError(context, 'Image base64 source requires a non-empty data string')
    }
    // Reject data-URL wrappers — callers must pass bare base64 + mediaType.
    if (source.data.startsWith('data:')) {
      throw formatError(context, 'Image base64 source must be raw base64, not a data URL')
    }
    if (!isBase64String(source.data)) {
      throw formatError(context, 'Image base64 source is not valid base64')
    }
    return
  }
  throw formatError(context, `Unsupported image source kind "${String((source as { kind?: unknown }).kind)}"`)
}

function isBase64String(value: string): boolean {
  // #236: accept only standard (or URL-safe) base64 with canonical padding.
  // Node Buffer.from is lenient (e.g. "AA=" decodes); reject non-canonical forms
  // via alphabet/length/padding checks plus decode → re-encode round-trip.
  if (value.length === 0 || value.length % 4 !== 0) return false
  if (!/^[A-Za-z0-9+/_-]+={0,2}$/.test(value)) return false
  // Padding may appear only as a suffix of length 0–2; no interior '='.
  const eq = value.indexOf('=')
  if (eq !== -1) {
    if (eq < value.length - 2) return false
    if (!/=+$/.test(value.slice(eq))) return false
  }
  try {
    const normalized = value.replace(/-/g, '+').replace(/_/g, '/')
    const buf = Buffer.from(normalized, 'base64')
    if (buf.length === 0) return false
    // Canonical re-encode must match the normalized alphabet exactly.
    if (buf.toString('base64') !== normalized) return false
    return true
  } catch {
    return false
  }
}


function formatError(
  context: { provider: string; model: string },
  message: string,
): ModelGatewayError {
  return new ModelGatewayError({
    code:      'MODEL_BAD_RESPONSE',
    message,
    phase:     'request',
    provider:  context.provider,
    model:     context.model,
    retryable: false,
  })
}
