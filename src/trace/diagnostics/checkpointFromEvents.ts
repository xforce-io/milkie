import type { Event } from '../types.js'
import type { AgentCheckpoint } from '../../types/store.js'

/**
 * The resume state for a run, projected from the event log: the payload of the
 * latest `agent.checkpoint` event. This makes the event log the single source
 * of truth for resume — no separate stateStore checkpoint blob required.
 * Returns null if the run was never checkpointed (it ran to completion).
 */
export function checkpointFromEvents(events: Event[]): AgentCheckpoint | null {
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i]!.type === 'agent.checkpoint') {
      return (events[i]!.payload as { checkpoint: AgentCheckpoint }).checkpoint
    }
  }
  return null
}
