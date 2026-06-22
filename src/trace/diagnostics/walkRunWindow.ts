import type { IEventStore } from '../EventStore.js'
import type { Event, AgentRunStartedPayload } from '../types.js'

/**
 * #189: walk the session run-chain newest→oldest from `startRunId`, bounded to
 * `lookback` runs. Mirrors Milkie.getSessionHistory's walk (each run's
 * agent.run.started.previousRunId links to the prior run) but bounded, so a
 * self-explain tool can read "my last N turns" without an external index.
 * Stops on undefined start, missing run (no events), or a cycle.
 */
export async function walkRunWindow(
  eventStore: IEventStore,
  startRunId: string | undefined,
  lookback: number,
): Promise<{ runId: string; events: Event[] }[]> {
  const out: { runId: string; events: Event[] }[] = []
  const seen = new Set<string>()
  let runId = startRunId
  while (runId && !seen.has(runId) && out.length < lookback) {
    seen.add(runId)
    const events = await eventStore.readByRunId(runId)
    if (events.length === 0) break
    out.push({ runId, events })
    const started = events.find(e => e.type === 'agent.run.started')
    runId = (started?.payload as AgentRunStartedPayload | undefined)?.previousRunId
  }
  return out
}
