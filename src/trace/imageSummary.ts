import { createHash } from 'crypto'
import type { MessageContent, ImageMediaType } from '../types/common.js'
import type { ModelRequest } from '../types/model.js'

export interface ImageSourceTraceSummary {
  kind: 'url' | 'base64'
  /** Redacted URL (no query/fragment) when kind=url. */
  url?: string
  /** SHA-256 hex of decoded bytes when kind=base64. */
  sha256?: string
  /** Decoded byte length when known. */
  byteLength?: number
}

export interface ImageContentTraceSummary {
  type: 'image'
  mediaType: ImageMediaType
  source: ImageSourceTraceSummary
}

/**
 * #236: strip credentials from image URLs before they hit the event log.
 * Keeps protocol/host/path; drops userinfo, query, and fragment (secret carriers).
 */
export function redactImageUrl(url: string): string {
  try {
    const u = new URL(url)
    u.username = ''
    u.password = ''
    u.search = ''
    u.hash = ''
    return u.toString()
  } catch {
    return '[invalid-url]'
  }
}

export function summarizeImageContent(
  part: Extract<MessageContent, { type: 'image' }>,
): ImageContentTraceSummary {
  if (part.source.kind === 'url') {
    return {
      type:      'image',
      mediaType: part.mediaType,
      source:    { kind: 'url', url: redactImageUrl(part.source.url) },
    }
  }

  const normalized = part.source.data.replace(/-/g, '+').replace(/_/g, '/')
  const buf = Buffer.from(normalized, 'base64')
  return {
    type:      'image',
    mediaType: part.mediaType,
    source:    {
      kind:       'base64',
      sha256:     createHash('sha256').update(buf).digest('hex'),
      byteLength: buf.length,
    },
  }
}

/**
 * Produce a trace-safe copy of a model request: image base64 payloads become
 * sha256/byteLength summaries; image URLs are redacted. Text/tool parts unchanged.
 * Return type is structural ModelRequest; image source shape is the audit summary.
 */
export function sanitizeModelRequestForTrace(request: ModelRequest): ModelRequest {
  const messages = request.messages.map(msg => ({
    role: msg.role,
    content: msg.content.map(part => {
      if (part.type === 'image') return summarizeImageContent(part) as unknown as MessageContent
      return part
    }),
  }))
  return { ...request, messages }
}
