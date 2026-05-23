import type { Message } from '../types/common.js'
import type { ModelRequest, ToolSchema } from '../types/model.js'
import type { WorkingMemory } from '../store/WorkingMemory.js'

export interface ContextLayerOptions {
  systemPrompt:        string
  model:               string
  maxHistoryMessages?: number
}

export class ContextLayer {
  private readonly systemPrompt: string
  private readonly model:        string
  private readonly maxHistory:   number

  private instructions:  Map<string, string> = new Map()
  private history:       Message[] = []
  private _currentTurn:  string | null = null
  private contextEpoch:  number = 0

  constructor(opts: ContextLayerOptions) {
    this.systemPrompt = opts.systemPrompt
    this.model        = opts.model
    this.maxHistory   = opts.maxHistoryMessages ?? 50
  }

  get currentTurn(): string | null { return this._currentTurn }

  setCurrentTurn(input: string): void {
    this._currentTurn = input
  }

  appendHistory(message: Message): void {
    this.history.push(message)
    if (this.history.length > this.maxHistory * 2) {
      this.history = this.history.slice(-this.maxHistory)
    }
  }

  loadInstructions(name: string, instructions: string): void {
    this.instructions.set(name, instructions)
    this.contextEpoch++
  }

  unloadInstructions(name: string): void {
    if (this.instructions.delete(name)) this.contextEpoch++
  }

  getContextEpoch(): number { return this.contextEpoch }
  getLoadedInstructions(): string[] { return Array.from(this.instructions.keys()) }

  buildRequest(
    tools:              ToolSchema[],
    workingMemory:      WorkingMemory,
    stateInstructions?: string,
  ): ModelRequest {
    // Build system string
    const systemParts: string[] = [this.systemPrompt]

    for (const [name, instr] of this.instructions.entries()) {
      systemParts.push(`\n--- Skill: ${name} ---\n${instr}`)
    }
    if (stateInstructions) {
      systemParts.push(`\n--- Current Instructions ---\n${stateInstructions}`)
    }
    const wmData = workingMemory.toJSON() as { data: Record<string, unknown>; log: unknown[] }
    if (Object.keys(wmData.data).length > 0 || wmData.log.length > 0) {
      systemParts.push('\n--- Working Memory ---\n' + JSON.stringify(wmData, null, 2))
    }

    // Build messages: history + current turn
    const messages: Message[] = [...this.history]
    if (this._currentTurn !== null) {
      messages.push({
        role:    'user',
        content: [{ type: 'text', text: this._currentTurn }],
      })
    }

    return {
      model:   this.model,
      system:  systemParts.join('\n'),
      messages,
      tools:   tools.length > 0 ? tools : undefined,
    }
  }

  snapshot(): {
    history:              Message[]
    instructionsSnapshot: string[]
    instructions?:        Record<string, string>
    contextEpoch:         number
  } {
    return {
      history:              [...this.history],
      instructionsSnapshot: this.getLoadedInstructions(),
      instructions:         Object.fromEntries(this.instructions.entries()),
      contextEpoch:         this.contextEpoch,
    }
  }

  restore(snapshot: {
    history:              Message[]
    instructionsSnapshot: string[]
    instructions?:        Record<string, string>
    contextEpoch:         number
  }): void {
    this.history      = snapshot.history
    this.contextEpoch = snapshot.contextEpoch
    this.instructions = new Map(Object.entries(snapshot.instructions ?? {}))
  }
}
