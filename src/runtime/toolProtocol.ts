import type { ToolCall } from '../types/tool.js'

export const CONTROL_TOOL_NAMES = new Set(['create_plan', 'update_step'])

export const TOOL_ARGUMENTS_INVALID_JSON = 'TOOL_ARGUMENTS_INVALID_JSON'
export const TOOL_ARGUMENTS_SCHEMA_INVALID = 'TOOL_ARGUMENTS_SCHEMA_INVALID'
export const TOOL_EXECUTION_ERROR = 'TOOL_EXECUTION_ERROR'

export interface ProtocolReject {
  ok: false
  code: typeof TOOL_ARGUMENTS_INVALID_JSON | typeof TOOL_ARGUMENTS_SCHEMA_INVALID
  message: string
}

export interface ProtocolAccept {
  ok: true
  input: unknown
}

/**
 * Bounded JSON repair: trailing commas, or a unique missing closer.
 * Ambiguous or unparseable → undefined (caller rejects).
 */
export function tryRepairJson(raw: string): unknown | undefined {
  if (typeof raw !== 'string' || raw.trim() === '') return undefined
  const stripped = raw.replace(/,\s*([}\]])/g, '$1')
  const unique = new Map<string, unknown>()
  const consider = (text: string) => {
    try {
      const value = JSON.parse(text) as unknown
      unique.set(JSON.stringify(value), value)
    } catch { /* not a unique parse */ }
  }
  consider(raw)
  consider(stripped)
  if (unique.size === 1) return unique.values().next().value
  unique.clear()
  for (const suffix of ['}', ']', ']}', '"}']) {
    consider(stripped + suffix)
  }
  if (unique.size === 1) return unique.values().next().value
  return undefined
}

export function validateControlToolInput(name: string, input: unknown): ProtocolAccept | ProtocolReject {
  if (name === 'create_plan') return validateCreatePlan(input)
  if (name === 'update_step') return validateUpdateStep(input)
  return { ok: true, input }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function validateCreatePlan(input: unknown): ProtocolAccept | ProtocolReject {
  if (!isPlainObject(input)) {
    return { ok: false, code: TOOL_ARGUMENTS_SCHEMA_INVALID, message: 'create_plan input must be an object' }
  }
  const extra = Object.keys(input).filter(k => k !== 'steps')
  if (extra.length > 0) {
    return { ok: false, code: TOOL_ARGUMENTS_SCHEMA_INVALID, message: `create_plan unexpected field: ${extra[0]}` }
  }
  if (!Array.isArray(input.steps) || input.steps.length < 1 || input.steps.some(s => typeof s !== 'string')) {
    return { ok: false, code: TOOL_ARGUMENTS_SCHEMA_INVALID, message: 'create_plan requires steps: non-empty string[]' }
  }
  return { ok: true, input }
}

function validateUpdateStep(input: unknown): ProtocolAccept | ProtocolReject {
  if (!isPlainObject(input)) {
    return { ok: false, code: TOOL_ARGUMENTS_SCHEMA_INVALID, message: 'update_step input must be an object' }
  }
  const extra = Object.keys(input).filter(k => k !== 'stepId' && k !== 'status')
  if (extra.length > 0) {
    return { ok: false, code: TOOL_ARGUMENTS_SCHEMA_INVALID, message: `update_step unexpected field: ${extra[0]}` }
  }
  if (typeof input.stepId !== 'number' || !Number.isFinite(input.stepId)) {
    return { ok: false, code: TOOL_ARGUMENTS_SCHEMA_INVALID, message: 'update_step requires stepId: number' }
  }
  if (input.status !== 'done' && input.status !== 'failed') {
    return { ok: false, code: TOOL_ARGUMENTS_SCHEMA_INVALID, message: 'update_step requires status: done | failed' }
  }
  return { ok: true, input }
}

/**
 * Resolve a control-tool call: optional in-memory raw repair, then narrow schema.
 * Does not invoke the handler.
 */
export function resolveControlToolCall(call: ToolCall): ProtocolAccept | ProtocolReject {
  let input = call.input
  if (call.invalidArguments) {
    const raw = call.rawArguments
    if (typeof raw !== 'string') {
      return {
        ok: false,
        code: TOOL_ARGUMENTS_INVALID_JSON,
        message: call.invalidArguments.message,
      }
    }
    const repaired = tryRepairJson(raw)
    if (repaired === undefined) {
      return {
        ok: false,
        code: TOOL_ARGUMENTS_INVALID_JSON,
        message: call.invalidArguments.message,
      }
    }
    input = repaired
  } else if (typeof input === 'string') {
    const repaired = tryRepairJson(input)
    if (repaired === undefined) {
      return { ok: false, code: TOOL_ARGUMENTS_INVALID_JSON, message: 'Tool arguments are not valid JSON' }
    }
    input = repaired
  }
  return validateControlToolInput(call.name, input)
}
