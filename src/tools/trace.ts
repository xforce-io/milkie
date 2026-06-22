import type { IEventStore } from '../trace/EventStore.js'
import type { ITraceObjectStore } from '../trace/TraceObjectStore.js'
import type { ToolDefinition, ToolContext } from '../types/tool.js'
import type { AgentRunStartedPayload, AgentRunCompletedPayload } from '../trace/types.js'
import { regionReuseCounts } from '../trace/RegionContextView.js'
import { buildExecutionProjection } from '../trace/diagnostics/buildExecutionProjection.js'
import type { ExecutionStep, ToolStep } from '../trace/diagnostics/buildExecutionProjection.js'
import { walkRunWindow } from '../trace/diagnostics/walkRunWindow.js'

/**
 * Read-Trace tools for the diagnoser agent: deterministic projections over a
 * recorded run's event log. Wrap core projections; never parse raw events in
 * the agent. `runId` is the run being DIAGNOSED (distinct from the diagnoser's
 * own run).
 */
export function makeTraceTools(
  eventStore: IEventStore,
  objectStore?: ITraceObjectStore,
): ToolDefinition[] {
  const get_run_io: ToolDefinition = {
    name: 'get_run_io',
    description: '取被诊断 run 的用户问题与最终答案。入参 { runId }。',
    inputSchema: { type: 'object', properties: { runId: { type: 'string' } }, required: ['runId'] },
    handler: async (input) => {
      const { runId } = input as { runId: string }
      const events = await eventStore.readByRunId(runId)
      let question = '', finalAnswer = ''
      for (const e of events) {
        if (e.type === 'agent.run.started') question = String((e.payload as AgentRunStartedPayload).input ?? '')
        if (e.type === 'agent.run.completed') finalAnswer = String((e.payload as AgentRunCompletedPayload).lastTextOutput ?? '')
      }
      return { question, finalAnswer }
    },
  }
  const LOOKBACK_DEFAULT = 3
  const LOOKBACK_MAX = 10

  /** Self view: drop llm step prompt/response bodies, keep tool steps. */
  function selfShape(steps: ExecutionStep[]): { toolSteps: ToolStep[]; llmStepCount: number } {
    const toolSteps = steps.filter(s => s.kind === 'tool' && s.tool).map(s => s.tool!) as ToolStep[]
    const llmStepCount = steps.filter(s => s.kind === 'llm').length
    return { toolSteps, llmStepCount }
  }

  async function regionContentFor(events: Awaited<ReturnType<typeof eventStore.readByRunId>>) {
    const regionContent = new Map<string, string>()
    if (objectStore) {
      for (const h of regionReuseCounts(events).keys()) {
        const c = await objectStore.getCanonical(h)
        if (c !== undefined) regionContent.set(h, c)
      }
    }
    return regionContent
  }

  const get_execution: ToolDefinition = {
    name: 'get_execution',
    description: '取执行投影:步骤序列(LLM/工具调用、工具 query、命中证据、region 组成)。' +
      '诊断:传 { runId } 取该 run 全量投影(steps)。自溯源:不传 runId,取自己最近 N 轮(默认 3)的工具步骤摘要(turns;不含 prompt 正文),可加 { lookback }。',
    inputSchema: { type: 'object', properties: {
      runId:    { type: 'string' },
      lookback: { type: 'number', description: '自溯源回看的轮数,默认 3,上限 10' },
    } },
    handler: async (input, ctx) => {
      const { runId, lookback } = (input ?? {}) as { runId?: string; lookback?: number }
      if (runId) {
        const events = await eventStore.readByRunId(runId)
        return buildExecutionProjection(events, { regionContent: await regionContentFor(events) })
      }
      const n = Math.max(1, Math.min(LOOKBACK_MAX, lookback ?? LOOKBACK_DEFAULT))
      const window = await walkRunWindow(eventStore, (ctx as ToolContext | undefined)?.previousRunId, n)
      const turns = []
      for (const { runId: rid, events } of window) {
        const proj = buildExecutionProjection(events, { regionContent: await regionContentFor(events) })
        turns.push({ runId: rid, ...selfShape(proj.steps) })
      }
      return { turns }
    },
  }
  return [get_run_io, get_execution]
}
