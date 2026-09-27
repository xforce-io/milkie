import type { AgentResult } from '../types/common.js'
import type { AgentRunCompletedPayload } from '../trace/types.js'

/** One projection for SDK results at every persisted or streamed terminal. */
export function completedPayload(result: AgentResult): AgentRunCompletedPayload {
  return {
    status: result.status,
    lastTextOutput: result.output,
    stopReason: result.stopReason,
    partial: result.partial,
    artifacts: result.artifacts,
    ...(result.stopCode ? { stopCode: result.stopCode } : {}),
    ...(result.checkpointId ? { checkpointId: result.checkpointId } : {}),
    ...(result.error ? { error: result.error } : {}),
  }
}
