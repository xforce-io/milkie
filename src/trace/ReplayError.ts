/**
 * Thrown when a replay cannot proceed for structural reasons:
 * missing lifecycle event, unknown agentId, malformed cached response,
 * empty event log, etc.
 *
 * Distinct from ReplayDivergenceError which fires on cache miss.
 */
export class ReplayError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ReplayError'
  }
}
