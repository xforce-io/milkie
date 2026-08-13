import type { ArtifactRef, StopReason } from '../types/common.js'
import type { Event } from './types.js'

export class TraceInspectError extends Error {
  readonly code = 'TRACE_INSPECT_INCOMPLETE' as const
  constructor(message: string) {
    super(message)
    this.name = 'TraceInspectError'
  }
}

export interface RunToolSummary {
  name: string
  count: number
  errorCount: number
}

export interface RunErrorSummary {
  code: string
  phase?: string
}

export interface RunSummary {
  runId: string
  status: 'completed' | 'interrupted' | 'error'
  stopReason?: StopReason
  stopCode?: string
  partial?: boolean
  checkpointId?: string
  turns: number
  tools: RunToolSummary[]
  errors: RunErrorSummary[]
  artifacts: ArtifactRef[]
}

export function parseJsonlEvents(content: string): Event[] {
  const lines = content.split('\n')
  const events: Event[] = []
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!
    if (line.length === 0) continue
    try {
      events.push(JSON.parse(line) as Event)
    } catch {
      throw new TraceInspectError(`JSONL line ${i + 1} is not valid JSON`)
    }
  }
  if (events.length === 0) {
    throw new TraceInspectError('run JSONL is empty or missing')
  }
  return events
}

export function summarizeRun(events: readonly Event[], runId?: string): RunSummary {
  if (events.length === 0) {
    throw new TraceInspectError('run has no events')
  }
  const id = runId ?? events[0]!.runId
  const completed = [...events].reverse().find(e => e.type === 'agent.run.completed')
  const payload = (completed?.payload ?? {}) as {
    status?: RunSummary['status']
    stopReason?: StopReason
    stopCode?: string
    partial?: boolean
    checkpointId?: string
    artifacts?: ArtifactRef[]
    error?: { code?: string; phase?: string } | string
  }

  const tools = new Map<string, RunToolSummary>()
  const errors: RunErrorSummary[] = []
  let turns = 0

  for (const e of events) {
    if (e.type === 'llm.responded') turns += 1
    if (e.type === 'tool.responded') {
      const p = e.payload as { toolName?: string; status?: string; error?: { code?: string; phase?: string } | string }
      const name = typeof p.toolName === 'string' ? p.toolName : '?'
      const row = tools.get(name) ?? { name, count: 0, errorCount: 0 }
      row.count += 1
      const isError = p.status === 'error' || p.error !== undefined
      if (isError) {
        row.errorCount += 1
        const code = typeof p.error === 'object' && p.error && 'code' in p.error
          ? String(p.error.code)
          : 'TOOL_ERROR'
        errors.push({
          code,
          ...(typeof p.error === 'object' && p.error && 'phase' in p.error && p.error.phase
            ? { phase: String(p.error.phase) }
            : {}),
        })
      }
      tools.set(name, row)
    }
  }

  if (payload.error) {
    if (typeof payload.error === 'string') {
      errors.push({ code: 'RUNTIME_ERROR' })
    } else if (payload.error.code) {
      errors.push({
        code: payload.error.code,
        ...(payload.error.phase ? { phase: payload.error.phase } : {}),
      })
    }
  }

  return {
    runId: id,
    status: payload.status ?? 'completed',
    ...(payload.stopReason ? { stopReason: payload.stopReason } : {}),
    ...(payload.stopCode ? { stopCode: payload.stopCode } : {}),
    ...(payload.partial !== undefined ? { partial: payload.partial } : {}),
    ...(payload.checkpointId ? { checkpointId: payload.checkpointId } : {}),
    turns,
    tools: [...tools.values()],
    errors,
    artifacts: Array.isArray(payload.artifacts) ? payload.artifacts : [],
  }
}
