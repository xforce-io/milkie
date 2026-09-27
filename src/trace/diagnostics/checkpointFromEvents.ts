import type { Event } from '../types.js'
import type { AgentCheckpoint } from '../../types/store.js'

/**
 * The resume state for a run, projected from the event log: the payload of the
 * latest `agent.checkpoint` event. This makes the event log the single source
 * of truth for resume — no separate stateStore checkpoint blob required.
 * An optional ID selects exactly that snapshot; never falls back to latest.
 * Returns null when no matching checkpoint was saved.
 */
export function checkpointFromEvents(events: Event[], checkpointId?: string): AgentCheckpoint | null {
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i]!.type === 'agent.checkpoint') {
      const checkpoint = (events[i]!.payload as { checkpoint: AgentCheckpoint }).checkpoint
      if (checkpointId === undefined || checkpoint.checkpointId === checkpointId) return checkpoint
    }
  }
  return null
}
