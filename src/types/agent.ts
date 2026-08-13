import type { BudgetFinalizeContext, DeliverableSpec } from './common.js'

export interface FSMState {
  name:            string
  type:            'llm' | 'action'
  instructions?:   string
  tools?:          string[]
  on?:             Record<string, string>
  handler?:        string
  terminal?:       boolean
  max_iterations?: number
}

export interface FSMDefinition {
  states: FSMState[]
  max_tool_calls?: number
}

export interface ModelConfig {
  provider: string
  model:    string
  adapter:  string
  baseUrl?: string
  options?: Record<string, unknown>
  /**
   * #236: explicit model/gateway capabilities. OpenAI-compatible adapters never
   * infer image support from endpoint/provider names — set imageInput: true only
   * when the deployed model accepts vision input.
   */
  capabilities?: {
    imageInput?: boolean
  }
}

/**
 * #235: stable Runtime-published built-in tool names (e.g. run_command).
 * extraTools / sub-agent names are not valid allowlist entries.
 */
export type BuiltinToolName = string

/** #235: allowlist of built-in tools this agent may expose. */
export interface BuiltinToolPolicy {
  allow: BuiltinToolName[]
}

export interface AgentConfig {
  agentId:      string
  version:      string
  systemPrompt: string
  fsm:          FSMDefinition
  model?:       ModelConfig
  /**
   * #126: open named model tiers. `model` stays the default tier; `models[tier]`
   * lets a one-shot `complete({ tier })` pick a named model/gateway. Keys are
   * arbitrary (milkie does not hardcode `default`/`fast`); an unknown or omitted
   * tier falls back to `model`.
   */
  models?:      Record<string, ModelConfig>
  toolboxes?:   Record<string, string>
  skills?:      Record<string, string>
  skillInstructions?: Record<string, string>
  subAgents?:   Record<string, string>
  dispatch?:    'local' | 'queue'
  /**
   * #235: built-in tool allowlist.
   * - omitted: compat mode — all current built-in tools are registered.
   * - empty allow: zero built-in tools (custom/extra tools still register).
   */
  builtinTools?: BuiltinToolPolicy
  /** #247: default deliverable contract when invoke omits `deliverables`. */
  deliverables?: DeliverableSpec[]
  /**
   * #244: optional hook after budget/deadline stop. Must not start LLM calls.
   * Failure is recorded as FINALIZE_FAILED and does not rewrite stopReason.
   */
  onBudgetFinalize?: (ctx: BudgetFinalizeContext) => Promise<void>
}
