import { createHash } from 'crypto'
import type { ModelRequest } from '../types/model.js'

/**
 * Canonical JSON serialization: object keys sorted recursively,
 * undefined values omitted, arrays preserve order. Used so two
 * structurally-equal payloads always produce the same hash.
 */
export function canonicalize(value: unknown): string {
  return JSON.stringify(normalize(value))
}

function normalize(value: unknown): unknown {
  if (value === null) return null
  if (Array.isArray(value)) return value.map(normalize)
  if (typeof value === 'object') {
    const out: Record<string, unknown> = {}
    const keys = Object.keys(value as object).sort()
    for (const k of keys) {
      const v = (value as Record<string, unknown>)[k]
      if (v === undefined) continue
      out[k] = normalize(v)
    }
    return out
  }
  return value
}

function sha256Hex(s: string): string {
  return createHash('sha256').update(s).digest('hex')
}

export function hashModelRequest(req: ModelRequest): string {
  return sha256Hex(canonicalize(req))
}

export function hashToolCall(toolName: string, input: unknown): string {
  return sha256Hex(canonicalize({ toolName, input }))
}
