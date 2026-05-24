import type { Event, AgentRunStartedPayload, AgentRunCompletedPayload } from './types.js'
import { ReplayError } from './ReplayError.js'

export interface RunSnapshot {
  agentId:        string
  goal:           string
  input:          string
  contextId:      string
  parentId?:      string
  terminalStatus?: AgentRunCompletedPayload['status']
}

/**
 * Pure projection: pulls the run's lifecycle identity from
 * agent.run.started and (optionally) agent.run.completed. Throws
 * ReplayError if the started event is missing — Phase 2 runs (no
 * lifecycle events) cannot be replayed.
 */
export function extractRunSnapshot(events: Event[]): RunSnapshot {
  if (events.length === 0) throw new ReplayError('no events for this run')

  const started = events.find(e => e.type === 'agent.run.started')
  if (!started) throw new ReplayError('no lifecycle start event; run was recorded before Phase 3')

  const startPayload = started.payload as AgentRunStartedPayload

  const completed = events.find(e => e.type === 'agent.run.completed')
  const terminalStatus = completed
    ? (completed.payload as AgentRunCompletedPayload).status
    : undefined

  return {
    agentId:        startPayload.agentId,
    goal:           startPayload.goal,
    input:          startPayload.input,
    contextId:      startPayload.contextId,
    parentId:       startPayload.parentId,
    terminalStatus,
  }
}
