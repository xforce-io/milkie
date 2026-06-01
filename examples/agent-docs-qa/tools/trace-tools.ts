import type { IEventStore } from '../../../src/trace/EventStore'
import type { ITraceObjectStore } from '../../../src/trace/TraceObjectStore'
import type { ToolDefinition } from '../../../src/types/tool'
import type { AgentRunStartedPayload, AgentRunCompletedPayload } from '../../../src/trace/types'
import { regionReuseCounts } from '../../../src/trace/RegionContextView'
import { buildExecutionProjection } from '../../../src/trace/diagnostics/buildExecutionProjection'

/**
 * Read-Trace tools for the diagnoser agent: deterministic projections over a
 * recorded run's event log. Wrap core projections; never parse raw events in
 * the agent. `runId` is the run being DIAGNOSED (distinct from the diagnoser's
 * own run).
 */
export function makeTraceTools(
  eventStore: IEventStore,
  objectStore: ITraceObjectStore,
): ToolDefinition[] {
  const get_run_io: ToolDefinition = {
    name: 'get_run_io',
    description: '取被诊断 run 的用户问题与最终答案。入参 { runId }。',
    inputSchema: { type: 'object', properties: { runId: { type: 'string' } }, required: ['runId'] },
    handler: async (input) => {
      const { runId } = input as { runId: string }
      const events = await eventStore.readByRunId(runId)
      let question = ''
      let finalAnswer = ''
      for (const e of events) {
        if (e.type === 'agent.run.started') question = String((e.payload as AgentRunStartedPayload).input ?? '')
        if (e.type === 'agent.run.completed') finalAnswer = String((e.payload as AgentRunCompletedPayload).lastTextOutput ?? '')
      }
      return { question, finalAnswer }
    },
  }

  const get_execution: ToolDefinition = {
    name: 'get_execution',
    description: '取被诊断 run 的执行投影:步骤序列(LLM/工具调用、工具 query、命中证据、region 组成)。入参 { runId }。',
    inputSchema: { type: 'object', properties: { runId: { type: 'string' } }, required: ['runId'] },
    handler: async (input) => {
      const { runId } = input as { runId: string }
      const events = await eventStore.readByRunId(runId)
      const regionContent = new Map<string, string>()
      for (const h of regionReuseCounts(events).keys()) {
        const c = await objectStore.getCanonical(h)
        if (c !== undefined) regionContent.set(h, c)
      }
      return buildExecutionProjection(events, { regionContent })
    },
  }

  return [get_run_io, get_execution]
}
