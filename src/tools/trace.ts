import type { IEventStore } from '../trace/EventStore.js'
import type { ITraceObjectStore } from '../trace/TraceObjectStore.js'
import type { ToolDefinition, ToolContext } from '../types/tool.js'
import type { AgentRunStartedPayload, AgentRunCompletedPayload } from '../trace/types.js'
import { regionReuseCounts } from '../trace/RegionContextView.js'
import { buildExecutionProjection } from '../trace/diagnostics/buildExecutionProjection.js'
import type { ExecutionStep, ToolStep } from '../trace/diagnostics/buildExecutionProjection.js'
import { walkRunWindow } from '../trace/diagnostics/walkRunWindow.js'
import { resolveClaimSources } from '../trace/diagnostics/buildLineageProjection.js'

/**
 * Read-Trace tools for the diagnoser agent: deterministic projections over a
 * recorded run's event log. Wrap core projections; never parse raw events in
 * the agent. `runId` is the run being DIAGNOSED (distinct from the diagnoser's
 * own run).
 */
/**
 * `selfOnly` (default false): restrict to self-溯源 only — drop the `runId` axis
 * from get_execution/get_lineage and omit get_run_io entirely. This is the GENERIC
 * registration handed to every eventStore-enabled agent (#196). Reading an arbitrary
 * `runId` is the diagnoser's privilege, granted via the full version through
 * loadStandardAgents(); on a shared serve instance the full version would let one
 * session read another's recorded I/O.
 */
export function makeTraceTools(
  eventStore: IEventStore,
  objectStore?: ITraceObjectStore,
  opts?: { selfOnly?: boolean },
): ToolDefinition[] {
  const selfOnly = opts?.selfOnly ?? false
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
  const LOOKBACK_MAX = 30

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
    description: selfOnly
      ? '自溯源执行投影:取自己最近 N 轮(默认 3)的工具步骤摘要(turns;不含 prompt 正文),可加 { lookback }。'
      : '取执行投影:步骤序列(LLM/工具调用、工具 query、命中证据、region 组成)。' +
        '诊断:传 { runId } 取该 run 全量投影(steps)。自溯源:不传 runId,取自己最近 N 轮(默认 3)的工具步骤摘要(turns;不含 prompt 正文),可加 { lookback }。',
    inputSchema: { type: 'object', properties: {
      ...(selfOnly ? {} : { runId: { type: 'string' } }),
      lookback: { type: 'number', description: '自溯源回看的轮数,默认 3,上限 30' },
    } },
    handler: async (input, ctx) => {
      const raw = (input ?? {}) as { runId?: string; lookback?: number }
      const runId = selfOnly ? undefined : raw.runId
      const lookback = raw.lookback
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
  const get_lineage: ToolDefinition = {
    name: 'get_lineage',
    description: selfOnly
      ? '溯源:某条结论/数字引用了哪条源。传 { query } 按结论文本子串匹配,在自己最近 N 轮(lookback,默认 3)里搜。返回 matches:{ runId, claim, sources }。'
      : '溯源:某条结论/数字引用了哪条源。传 { query } 按结论文本子串匹配(如对话里出现的数字),' +
        '默认在自己最近 N 轮(lookback,默认 3)里搜;也可传 { runId } 限定单轮。返回 matches:{ runId, claim, sources }。',
    inputSchema: { type: 'object', properties: {
      ...(selfOnly ? {} : { runId: { type: 'string' } }),
      lookback: { type: 'number', description: '回看轮数,默认 3,上限 30' },
      query:    { type: 'string', description: '要溯源的结论/数字文本' },
    } },
    handler: async (input, ctx) => {
      const raw = (input ?? {}) as { runId?: string; lookback?: number; query?: string }
      const runId = selfOnly ? undefined : raw.runId
      const { lookback, query } = raw
      const window = runId
        ? [{ runId, events: await eventStore.readByRunId(runId) }]
        : await walkRunWindow(eventStore, (ctx as ToolContext | undefined)?.previousRunId,
            Math.max(1, Math.min(LOOKBACK_MAX, lookback ?? LOOKBACK_DEFAULT)))
      const matches = []
      for (const { runId: rid, events } of window) {
        for (const m of resolveClaimSources(events, query)) {
          matches.push({ runId: rid, claim: m.claim, sources: m.sources })
        }
      }
      return { matches }
    },
  }
  return selfOnly ? [get_execution, get_lineage] : [get_run_io, get_execution, get_lineage]
}
