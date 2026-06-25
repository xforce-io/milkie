import type { IEventStore } from '../trace/EventStore.js'
import type { ITraceObjectStore } from '../trace/TraceObjectStore.js'
import type { ToolDefinition, ToolContext } from '../types/tool.js'
import type { AgentRunStartedPayload, AgentRunCompletedPayload } from '../trace/types.js'
import { regionReuseCounts } from '../trace/RegionContextView.js'
import { buildExecutionProjection } from '../trace/diagnostics/buildExecutionProjection.js'
import type { ExecutionStep, ToolStep } from '../trace/diagnostics/buildExecutionProjection.js'
import { walkRunWindow } from '../trace/diagnostics/walkRunWindow.js'
import { resolveClaimSources, derefObject } from '../trace/diagnostics/buildLineageProjection.js'

/**
 * Read-Trace tools for the diagnoser agent: deterministic projections over a
 * recorded run's event log. Wrap core projections; never parse raw events in
 * the agent. `runId` is the run being DIAGNOSED (distinct from the diagnoser's
 * own run).
 */
/**
 * `selfOnly` (default false): the GENERIC registration handed to every
 * eventStore-enabled agent (#196). It withholds the diagnoser privilege of reading
 * an *arbitrary* runId — on a shared serve instance that would leak another session's
 * recorded I/O. It does NOT, however, blanket-drop the runId axis: a selfOnly agent
 * may dereference a runId that was *delivered to it* via a projection (#200 C). The
 * allowlist is `ctx.deliveredRunIds`; a runId outside it is ignored (get_execution /
 * get_lineage fall back to the self window) or refused (get_run_io). The full version,
 * granted via loadStandardAgents(), honours any runId.
 */
export function makeTraceTools(
  eventStore: IEventStore,
  objectStore?: ITraceObjectStore,
  opts?: { selfOnly?: boolean },
): ToolDefinition[] {
  const selfOnly = opts?.selfOnly ?? false

  /**
   * #200 C: resolve the runId a selfOnly handler may act on. Full mode → any runId.
   * selfOnly → the runId only if it was delivered to this run (capability-by-handle);
   * otherwise undefined, so the caller treats it as "no runId" (self window).
   */
  function gatedRunId(rawRunId: string | undefined, ctx: ToolContext | undefined): string | undefined {
    if (!selfOnly) return rawRunId
    if (!rawRunId) return undefined
    return ctx?.deliveredRunIds?.includes(rawRunId) ? rawRunId : undefined
  }

  const get_run_io: ToolDefinition = {
    name: 'get_run_io',
    description: selfOnly
      ? '取投递给你的某条产出 run 的用户问题与最终答案。入参 { runId },runId 必须是上下文里 producedBy.runId(投递给你的那条 run);未投递的 runId 会被拒绝。'
      : '取被诊断 run 的用户问题与最终答案。入参 { runId }。',
    inputSchema: { type: 'object', properties: { runId: { type: 'string' } }, required: ['runId'] },
    handler: async (input, ctx) => {
      const rawRunId = (input as { runId: string }).runId
      const runId = gatedRunId(rawRunId, ctx as ToolContext | undefined)
      if (!runId) return { error: `runId '${rawRunId}' 未在投递上下文中,拒绝解引用;只能解引用上下文里 producedBy.runId 的那条 run。` }
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
      ? '执行投影。不传 runId:取自己最近 N 轮(默认 3)的工具步骤摘要(turns;不含 prompt 正文),可加 { lookback }。' +
        '传 { runId }:仅当它是投递给你的 producedBy.runId 时,取该 run 全量投影(steps);未投递的 runId 被忽略,退回自溯源窗口。'
      : '取执行投影:步骤序列(LLM/工具调用、工具 query、命中证据、region 组成)。' +
        '诊断:传 { runId } 取该 run 全量投影(steps)。自溯源:不传 runId,取自己最近 N 轮(默认 3)的工具步骤摘要(turns;不含 prompt 正文),可加 { lookback }。',
    inputSchema: { type: 'object', properties: {
      runId: { type: 'string', ...(selfOnly ? { description: '投递给你的 producedBy.runId;未投递的 runId 被忽略' } : {}) },
      lookback: { type: 'number', description: '自溯源回看的轮数,默认 3,上限 30' },
    } },
    handler: async (input, ctx) => {
      const raw = (input ?? {}) as { runId?: string; lookback?: number }
      const runId = gatedRunId(raw.runId, ctx as ToolContext | undefined)
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
    description: '溯源某条结论/数字引用了哪条源。两种入口:' +
      '①精确解引用(推荐):传 { claimId } 或源 { objectId }(工具/上下文给你的真实 id),确定性返回其 cites 邻域(cites/citedBy),返回 refs,无文本匹配。' +
      '②弱检索:传 { query } 按结论文本子串匹配(如对话里出现的数字),返回 matches。' +
      (selfOnly
        ? '范围默认自己最近 N 轮(lookback,默认 3);也可传投递给你的 { runId }(未投递的 runId 被忽略)。'
        : '范围默认自己最近 N 轮(lookback,默认 3);也可传 { runId } 限定单轮。'),
    inputSchema: { type: 'object', properties: {
      runId:    { type: 'string', ...(selfOnly ? { description: '投递给你的 producedBy.runId;未投递的 runId 被忽略' } : {}) },
      lookback: { type: 'number', description: '回看轮数,默认 3,上限 30' },
      claimId:  { type: 'string', description: '精确解引用:claim 的 objectId,返回其引用的源(cites)' },
      objectId: { type: 'string', description: '精确解引用:源 object 的 objectId,返回引用它的结论(citedBy)' },
      query:    { type: 'string', description: '弱检索:要溯源的结论/数字文本(子串匹配,非精确入口)' },
    } },
    handler: async (input, ctx) => {
      const raw = (input ?? {}) as { runId?: string; lookback?: number; query?: string; claimId?: string; objectId?: string }
      const runId = gatedRunId(raw.runId, ctx as ToolContext | undefined)
      const { lookback, query } = raw
      const window = runId
        ? [{ runId, events: await eventStore.readByRunId(runId) }]
        : await walkRunWindow(eventStore, (ctx as ToolContext | undefined)?.previousRunId,
            Math.max(1, Math.min(LOOKBACK_MAX, lookback ?? LOOKBACK_DEFAULT)))
      // #200 A: exact handle dereference takes precedence over the weak substring query.
      const handle = raw.claimId ?? raw.objectId
      if (handle) {
        const refs = []
        for (const { runId: rid, events } of window) {
          const ref = derefObject(events, handle)
          if (ref) refs.push({ runId: rid, ...ref })
        }
        return { refs }
      }
      const matches = []
      for (const { runId: rid, events } of window) {
        for (const m of resolveClaimSources(events, query)) {
          matches.push({ runId: rid, claim: m.claim, sources: m.sources })
        }
      }
      return { matches }
    },
  }
  // #200 C: get_run_io is registered in BOTH modes; in selfOnly it is gated to
  // delivered runIds (refusing any other), so it carries no cross-session leak.
  return [get_run_io, get_execution, get_lineage]
}
