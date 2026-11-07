import { createHash, randomBytes, randomUUID } from 'node:crypto'
import { appendFileSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { ExecutionError, type ExecutionRecord, type ExecutionStatus } from './types.js'
import { ExecutionStore } from './store.js'

export interface ObservationOptions { traceparent: string; maxEvents: number; maxBytes: number }
export interface ObservationCapabilities {
  version: 1
  purge: true
  lifecycle: 'supervisor'
  activity: boolean
  usage: 'native-session-cumulative' | 'native-message' | 'gateway-response' | 'unavailable'
  modelRequests: 'gateway-boundary' | 'unavailable'
  hostTools: boolean
  hiddenReasoning: 'unavailable'
}
export function observationCapabilities(runtime: string): ObservationCapabilities {
  return { version: 1, purge: true, lifecycle: 'supervisor', activity: runtime !== 'api', usage: runtime === 'codex' ? 'native-session-cumulative' : runtime === 'pi' ? 'native-message' : runtime === 'api' ? 'gateway-response' : 'unavailable', modelRequests: runtime === 'api' ? 'gateway-boundary' : 'unavailable', hostTools: runtime !== 'api', hiddenReasoning: 'unavailable' }
}
export function assertObservation(value: ObservationOptions): void {
  if (!value || typeof value !== 'object' || Object.keys(value).some(k => !['traceparent', 'maxEvents', 'maxBytes'].includes(k)) || typeof value.traceparent !== 'string' || !/^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$/.test(value.traceparent) || /^0+$/.test(value.traceparent.slice(3, 35)) || /^0+$/.test(value.traceparent.slice(36, 52)) || !Number.isSafeInteger(value.maxEvents) || value.maxEvents < 2 || value.maxEvents > 100000 || !Number.isSafeInteger(value.maxBytes) || value.maxBytes < 4096 || value.maxBytes > 64 * 1024 * 1024) throw new ExecutionError('unsupported_constraint')
}
export interface ContentSummary { bytes: number; sha256: string }
export function contentSummary(value: unknown): ContentSummary {
  const text = typeof value === 'string' ? value : JSON.stringify(value) ?? 'undefined'
  return { bytes: Buffer.byteLength(text), sha256: createHash('sha256').update(text).digest('hex') }
}
export interface NativeCorrelation { sessionId?: string; turnId?: string; callId?: string; responseId?: string }
export interface UsageObservation {
  scope: 'native_session' | 'message' | 'gateway'
  aggregation: 'cumulative' | 'delta'
  inputTokens?: number
  outputTokens?: number
  totalTokens?: number
  cachedInputTokens?: number
  reasoningOutputTokens?: number
}
export interface ObservationPayloads {
  'execution.started': { runtime: string; capabilities: ObservationCapabilities }
  'execution.finished': { status: ExecutionStatus; stopped: boolean; code?: string }
  'execution.interrupted': { reason: 'cancelled' | 'timed_out' | 'host_lost' }
  'agent.activity': { category: 'turn' | 'message' | 'reasoning' | 'tool' | 'compaction' }
  'model.requested': { scope: 'gateway-boundary' }
  'model.responded': { scope: 'gateway-boundary'; outcome: 'succeeded' | 'failed' }
  usage: UsageObservation
  'tool.requested': { callId: string; name: string; input: ContentSummary }
  'tool.rejected': { callId: string; code: 'invalid_input' | 'rejected' }
  'tool.host.started': { callId: string }
  'tool.host.responded': { callId: string; outcome: 'succeeded' | 'invalid_input' | 'rejected' | 'execution_failed'; output?: ContentSummary }
  'tool.delivery': { callId: string; outcome: 'delivered' | 'unknown' }
  'tool.interrupted': { callId: string; reason: 'cancelled' | 'timed_out' | 'host_lost' | 'execution_ended' }
}
export type ObservationKind = keyof ObservationPayloads
export interface ExecutionObservation {
  version: 1
  id: string
  sourceId: string
  sequence: number
  contextId: string
  runId: string
  timestamp: number
  monotonicNs: string
  traceId: string
  spanId: string
  parentSpanId: string
  kind: ObservationKind
  native?: NativeCorrelation
  data: ObservationPayloads[ObservationKind]
}
export interface ObservationState {
  version: 1
  lastSequence: number
  recorded: number
  bytes: number
  dropped: number
  finished: boolean
  fault?: 'capacity' | 'storage'
}
export interface ObservationPage {
  enabled: boolean
  events: ExecutionObservation[]
  nextSequence: number
  status: { state: 'disabled' | 'pending' | 'recording' | 'finished' | 'incomplete'; complete: boolean; lastSequence: number; dropped: number; fault?: 'capacity' | 'storage' | 'corrupt' | 'unavailable' | 'expired' | 'cleanup_pending' }
}

/** One worker writes one bounded source. It never owns the business outcome. */
export class ObservationJournal {
  private readonly state: ObservationState = { version: 1, lastSequence: 0, recorded: 0, bytes: 0, dropped: 0, finished: false }
  private readonly agentSpan = randomBytes(8).toString('hex')
  private readonly traceId: string
  private readonly parentSpan: string
  private readonly flags: string
  private readonly file: string
  private readonly pendingTools = new Map<string, NativeCorrelation | undefined>()
  constructor(private readonly store: ExecutionStore, private readonly run: ExecutionRecord, private readonly options: ObservationOptions) {
    assertObservation(options)
    this.traceId = options.traceparent.slice(3, 35); this.parentSpan = options.traceparent.slice(36, 52); this.flags = options.traceparent.slice(53)
    this.file = join(store.root, 'observations', `${run.runId}.jsonl`)
    try {
      mkdirSync(join(store.root, 'observations'), { recursive: true, mode: 0o700 })
      writeFileSync(this.file, '', { mode: 0o600, flag: 'wx' })
      this.save()
    } catch { this.state.fault = 'storage' }
  }
  toolTrace(callId: string): string {
    const span = createHash('sha256').update(`${this.run.runId}:${callId}`).digest('hex').slice(0, 16)
    return `00-${this.traceId}-${span === '0000000000000000' ? '0000000000000001' : span}-${this.flags}`
  }
  finish(status: ExecutionStatus, stopped: boolean, code?: string, reason?: 'cancelled' | 'timed_out' | 'host_lost'): void {
    for (const [callId, native] of this.pendingTools) {
      this.emit('tool.interrupted', { callId, reason: reason ?? 'execution_ended' }, native, callId)
      this.emit('tool.delivery', { callId, outcome: 'unknown' }, native, callId)
    }
    this.emit('execution.finished', { status, stopped, ...(code ? { code } : {}) })
  }
  private save(): void {
    try { this.store.write('observations', this.run.runId, this.state) } catch { this.state.fault = 'storage' }
  }
  emit<K extends ObservationKind>(kind: K, data: ObservationPayloads[K], native?: NativeCorrelation, callId?: string): void {
    if (this.state.finished) return
    if (kind === 'tool.requested') this.pendingTools.set((data as ObservationPayloads['tool.requested']).callId, native)
    if (kind === 'tool.delivery') this.pendingTools.delete((data as ObservationPayloads['tool.delivery']).callId)
    const sequence = ++this.state.lastSequence
    const span = callId ? this.toolTrace(callId).slice(36, 52) : this.agentSpan
    const safeNative: NativeCorrelation = {}
    for (const k of ['sessionId', 'turnId', 'callId', 'responseId'] as const) if (native?.[k] && native[k]!.length <= 128) safeNative[k] = native[k]
    const event: ExecutionObservation = { version: 1, id: randomUUID(), sourceId: this.run.runId, sequence, runId: this.run.runId, contextId: this.run.contextId, timestamp: Date.now(), monotonicNs: process.hrtime.bigint().toString(), traceId: this.traceId, spanId: span, parentSpanId: callId ? this.agentSpan : this.parentSpan, kind, ...(Object.keys(safeNative).length ? { native: safeNative } : {}), data }
    const line = JSON.stringify(event) + '\n', bytes = Buffer.byteLength(line)
    const ending = kind === 'execution.finished'
    const limit = ending ? this.options.maxBytes : this.options.maxBytes - 1024
    if (this.state.fault === 'storage' || (!ending && this.state.recorded >= this.options.maxEvents - 1) || this.state.bytes + bytes > limit) {
      this.state.dropped++
      this.state.fault ??= 'capacity'
    } else {
      try { appendFileSync(this.file, line, { encoding: 'utf8' }); this.state.recorded++; this.state.bytes += bytes }
      catch { this.state.dropped++; this.state.fault = 'storage' }
    }
    if (ending) this.state.finished = true
    this.save()
  }
}

/** Parsing is intentionally conservative: a missing terminal or corrupt tail is incomplete. */
export function readObservations(store: ExecutionStore, run: ExecutionRecord, afterSequence = 0, limit = 100): ObservationPage {
  if (!Number.isSafeInteger(afterSequence) || afterSequence < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > 1000) throw new ExecutionError('invalid_request')
  if (!run.observation) return { enabled: false, events: [], nextSequence: afterSequence, status: { state: 'disabled', complete: false, lastSequence: 0, dropped: 0 } }
  if (run.observationCleanup) return { enabled: true, events: [], nextSequence: afterSequence, status: { state: 'incomplete', complete: false, lastSequence: 0, dropped: 0, fault: run.observationCleanup.completedAt ? 'expired' : 'cleanup_pending' } }
  const events: ExecutionObservation[] = []
  let state: ObservationState | undefined, fault: ObservationPage['status']['fault'], previous = 0, terminal = false
  try {
    state = store.read<ObservationState>('observations', run.runId)
    const file = join(store.root, 'observations', `${run.runId}.jsonl`)
    if (statSync(file).size > 64 * 1024 * 1024) throw new Error('oversized')
    const text = readFileSync(file, 'utf8')
    if (text && !text.endsWith('\n')) fault = 'corrupt'
    const ids = new Set<string>()
    for (const line of text.split('\n')) {
      if (!line) continue
      const e = JSON.parse(line) as ExecutionObservation
      if (e.version !== 1 || e.sourceId !== run.runId || e.runId !== run.runId || e.contextId !== run.contextId || !Number.isSafeInteger(e.sequence) || e.sequence <= previous || typeof e.id !== 'string' || ids.has(e.id) || e.traceId !== run.observation.traceparent.slice(3, 35)) throw new Error('corrupt')
      ids.add(e.id); previous = e.sequence; terminal = e.kind === 'execution.finished'
      if (e.sequence > afterSequence && events.length < limit) events.push(e)
    }
    if ((!state || state.lastSequence < previous || state.recorded !== ids.size || state.bytes !== Buffer.byteLength(text)) && (state?.finished || run.finishedAt)) fault ??= 'corrupt'
  } catch (e) { fault ??= (e as NodeJS.ErrnoException).code === 'ENOENT' ? 'unavailable' : 'corrupt' }
  const complete = !fault && !!state?.finished && terminal && !state.dropped && !state.fault
  const pending = !state && !run.finishedAt && fault === 'unavailable'
  return { enabled: true, events, nextSequence: events.at(-1)?.sequence ?? afterSequence, status: { state: pending ? 'pending' : fault || state?.fault || ((run.finishedAt || run.status === 'unknown') && !complete) ? 'incomplete' : complete ? 'finished' : 'recording', complete, lastSequence: Math.max(previous, state?.lastSequence ?? 0), dropped: state?.dropped ?? 0, ...(!pending && (fault || state?.fault) ? { fault: fault ?? state?.fault } : {}) } }
}


export function reportedUsage(raw: unknown, scope: UsageObservation['scope'], aggregation: UsageObservation['aggregation'], keys: Partial<Record<keyof Omit<UsageObservation, 'scope' | 'aggregation'>, string>>): UsageObservation | undefined {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return undefined
  const value = raw as Record<string, unknown>
  const result: UsageObservation = { scope, aggregation }
  let count = 0
  for (const [field, key] of Object.entries(keys)) {
    const n = value[key]
    if (typeof n === 'number' && Number.isSafeInteger(n) && n >= 0) { (result as unknown as Record<string, unknown>)[field] = n; count++ }
  }
  return count ? result : undefined
}
