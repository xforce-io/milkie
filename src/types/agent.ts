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
  toolboxes?:   Record<string, string>
  skills?:      Record<string, string>
  skillInstructions?: Record<string, string>
  subAgents?:   Record<string, string>
  stateStore?:  'memory' | 'sqlite' | 'redis'
  dispatch?:    'local' | 'queue'
}
