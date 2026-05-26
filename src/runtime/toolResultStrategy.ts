import type { Shape } from '../types/tool'

/**
 * Serialize raw tool output to a string for shaping + LLM consumption.
 * Mirrors the existing AgentRuntime.executeTools serialization
 * (JSON.stringify for objects, string passthrough, etc) so behavior
 * is unchanged when shape='verbatim'.
 */
export function serializeOutput(raw: unknown): string {
  if (raw === undefined) return ''
  if (typeof raw === 'string') return raw
  return JSON.stringify(raw)
}

/**
 * Apply a Shape to raw tool output (handler result OR error message).
 * Pure: same (raw, shape) → same string. Replay-safe.
 */
export function applyShape(raw: unknown, shape: Shape): string {
  const serialized = serializeOutput(raw)

  if (shape === 'verbatim') return serialized

  if (shape.kind === 'truncate') {
    if (serialized.length <= shape.maxChars) return serialized
    const head = serialized.slice(0, shape.maxChars)
    const droppedChars = serialized.length - shape.maxChars
    if (shape.tailHint) {
      return `${head}[...truncated ${droppedChars} chars...]`
    }
    return `${head}...`
  }

  if (shape.kind === 'tail') {
    if (serialized.length <= shape.maxChars) return serialized
    const tail = serialized.slice(-shape.maxChars)
    const droppedChars = serialized.length - shape.maxChars
    return `[...${droppedChars} chars dropped...]${tail}`
  }

  // Exhaustiveness check — TypeScript will error if a Shape variant is added without handling
  const _exhaustive: never = shape
  return serializeOutput(raw)
}
