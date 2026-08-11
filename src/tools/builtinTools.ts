import type { AgentConfig, BuiltinToolName, BuiltinToolPolicy } from '../types/agent.js'
import type { ToolDefinition } from '../types/tool.js'
import { cognitiveTools } from './cognitive.js'
import { execTools } from './exec.js'
import { lineageTools } from './lineage.js'
import { systemTools } from './system.js'

/**
 * #235: stable built-in tool identifiers published by Runtime.
 * `extraTools` and sub-agent tool names are intentionally excluded.
 */
export const BUILTIN_TOOL_NAMES = [
  'skill_list',
  'skill_request',
  'think',
  'create_plan',
  'update_step',
  'cite',
  'declare_relation',
  'run_command',
] as const satisfies readonly BuiltinToolName[]

const BUILTIN_NAME_SET = new Set<string>(BUILTIN_TOOL_NAMES)

/** All currently published built-in tool definitions, in stable registration order. */
export function allBuiltinToolDefinitions(): ToolDefinition[] {
  return [...systemTools, ...cognitiveTools, ...lineageTools, ...execTools]
}

export function isBuiltinToolName(name: string): name is BuiltinToolName {
  return BUILTIN_NAME_SET.has(name)
}

/**
 * Validate a declared policy. Unknown or duplicate names make the config invalid
 * before the run starts (no silent fallback to full authorization).
 */
export function validateBuiltinToolPolicy(policy: BuiltinToolPolicy): BuiltinToolName[] {
  if (!policy || !Array.isArray(policy.allow)) {
    throw new Error('AgentConfig.builtinTools.allow must be an array of built-in tool names')
  }

  const seen = new Set<string>()
  const allow: BuiltinToolName[] = []
  for (const raw of policy.allow) {
    if (typeof raw !== 'string' || raw.length === 0) {
      throw new Error('AgentConfig.builtinTools.allow entries must be non-empty strings')
    }
    if (!isBuiltinToolName(raw)) {
      throw new Error(
        `Unknown built-in tool name in AgentConfig.builtinTools.allow: "${raw}". ` +
        `Known names: ${BUILTIN_TOOL_NAMES.join(', ')}`,
      )
    }
    if (seen.has(raw)) {
      throw new Error(`Duplicate built-in tool name in AgentConfig.builtinTools.allow: "${raw}"`)
    }
    seen.add(raw)
    allow.push(raw)
  }
  return allow
}

/**
 * Resolve the effective built-in allowlist for a run.
 * - Root, policy omitted → all current built-ins (compat mode).
 * - Root, empty allow → zero built-ins.
 * - Child, policy omitted → inherit parent effective set.
 * - Child, policy set → parent effective ∩ child allow.
 */
export function resolveEffectiveBuiltinTools(
  config: Pick<AgentConfig, 'builtinTools'>,
  parentEffective?: readonly BuiltinToolName[],
): BuiltinToolName[] {
  const parent = parentEffective ? [...parentEffective] : [...BUILTIN_TOOL_NAMES]
  const parentSet = new Set<string>(parent)

  if (!config.builtinTools) {
    return parent.filter(isBuiltinToolName)
  }

  const allow = validateBuiltinToolPolicy(config.builtinTools)
  return allow.filter(name => parentSet.has(name))
}

/** Filter the published built-in definitions down to an effective allowlist. */
export function selectBuiltinToolDefinitions(effective: readonly BuiltinToolName[]): ToolDefinition[] {
  const allow = new Set<string>(effective)
  return allBuiltinToolDefinitions().filter(t => allow.has(t.name))
}
