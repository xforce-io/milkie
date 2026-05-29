/**
 * CausalCursor — per-run, mutable state threaded through the single run's
 * RecordingIOPort and AgentRuntime so each emitted trace event can stamp a
 * `causedBy` pointing at its upstream cause (#30, diagnosable P0).
 *
 * Pure state, no logic, no IO. One instance per run (root and each sub-agent
 * run get their own — see Milkie.buildMakeChildPort), so parent/child causal
 * chains never bleed into each other.
 *
 * INVARIANT (critical): every read/write of these fields MUST happen at the
 * synchronous moment the event is constructed — never inside a deferred
 * `enqueueTraceWrite` closure. Half the trace events append asynchronously, so
 * append order != causal order; capturing the cause value synchronously is the
 * only way the `causedBy` reflects the real predecessor (see spec §F1).
 */
export class CausalCursor {
  /** Most recent llm/tool event id emitted by RecordingIOPort. fsm.transition reads this. */
  lastIoEventId?: string
  /** Most recent llm.responded event id. tool.requested reads this (the frame that decided the call). */
  lastLlmRespondedId?: string
  /** Most recent turn-terminating event id (tool.responded, or agent.run.started seed). llm.requested reads this. */
  lastTerminatorId?: string
}
