/**
 * Task outcome — task-level judgment independent of execution status.
 * See ARCHITECTURE.md Outcome concept and invariant 14; story s-016.
 */

export type TaskOutcomeValue = 'success' | 'failure' | 'partial' | 'unknown'

/** Who recorded the judgment (eval harness, human, rule, business callback, …). */
export type TaskOutcomeSource = string

export interface TaskOutcomeScore {
  name:  string
  value: number | string | boolean
}

export interface RecordTaskOutcomeInput {
  runId:   string
  value:   TaskOutcomeValue
  /** Non-empty; recommended: eval | human | rule | business */
  source:  TaskOutcomeSource
  note?:   string
  scores?: TaskOutcomeScore[]
}

/** Query view of the latest outcome recorded for a run. */
export interface TaskOutcome {
  runId:      string
  value:      TaskOutcomeValue
  source:     TaskOutcomeSource
  recordedAt: number
  note?:      string
  scores?:    TaskOutcomeScore[]
}

export class TaskOutcomeRunNotFoundError extends Error {
  readonly runId: string
  constructor(runId: string) {
    super(`No run found for runId "${runId}" (no events in event store)`)
    this.name  = 'TaskOutcomeRunNotFoundError'
    this.runId = runId
  }
}

export class TaskOutcomeError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'TaskOutcomeError'
  }
}
