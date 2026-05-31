import { v4 as uuid } from 'uuid'
import type { IStateStore, AgentCheckpoint } from '../types/store.js'

export class CheckpointManager {
  constructor(
    private readonly store: IStateStore,
    private readonly agentId: string,
    private readonly agentRunId: string,
  ) {}

  private key(suffix: string): string {
    return `agent:${this.agentId}:run:${this.agentRunId}:checkpoint:${suffix}`
  }

  async save(checkpoint: Omit<AgentCheckpoint, 'checkpointId'>): Promise<AgentCheckpoint> {
    const full: AgentCheckpoint = {
      ...checkpoint,
      checkpointId: uuid(),
    }
    await this.store.set(this.key(full.checkpointId), full)
    await this.store.set(this.key('latest'), full)
    return full
  }

  async loadLatest(): Promise<AgentCheckpoint | null> {
    const raw = await this.store.get(this.key('latest'))
    return (raw as AgentCheckpoint | undefined) ?? null
  }

  async load(checkpointId: string): Promise<AgentCheckpoint | null> {
    const raw = await this.store.get(this.key(checkpointId))
    return (raw as AgentCheckpoint | undefined) ?? null
  }

  async getByContextAndTurn(contextId: string, turn: number): Promise<AgentCheckpoint | null> {
    const raw = await this.store.get(`context:${contextId}:turn:${turn}`)
    return (raw as AgentCheckpoint | undefined) ?? null
  }

  async saveForContext(contextId: string, turn: number, checkpoint: AgentCheckpoint): Promise<void> {
    await this.store.set(`context:${contextId}:turn:${turn}`, checkpoint)
  }
}
