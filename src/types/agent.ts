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
}

export interface ModelConfig {
  provider: string
  model:    string
  adapter:  string
  baseUrl?: string
  options?: Record<string, unknown>
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
}
