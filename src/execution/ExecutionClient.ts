import { fork, type ChildProcess } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { join } from 'node:path'
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { setTimeout as delay } from 'node:timers/promises'
import type { ConnectionInput } from '../connection/types.js'
import { resolveAndParseConnection } from '../connection/parse.js'
import { ExecutionStore, workingDirectory } from './store.js'
import { assertNativeSession, prepareCliStorage, resolveCliStorage, supervisorEnvironment } from './adapters.js'
import { assertHostTools, normalizeToolResult } from './hostTools.js'
import { ExecutionError, type CliStorage, type ExecutionCapabilities, type ExecutionClientOptions, type ExecutionConstraints, type ExecutionContext, type ExecutionRecord, type HostToolSpec, type ToolCallRecord, type ToolHandler, type WorkerRequest, type WorkerToolMessage } from './types.js'

const ACTIVE = new Set(['starting', 'running'])
export class ExecutionClient {
  private readonly store: ExecutionStore
  private readonly projection
  private readonly env: NodeJS.ProcessEnv
  private readonly connection: ConnectionInput
  /** Host-lifetime pipes. Closing one tells the worker to stop the CLI. */
  private readonly parentPipes: unknown[] = []
  constructor(options: ExecutionClientOptions) {
    this.connection = structuredClone(options.connection)
    this.projection = resolveAndParseConnection(this.connection).projection
    this.store = new ExecutionStore(options.dataDir)
    this.env = { ...(options.env ?? process.env) }
  }
  capabilities(): ExecutionCapabilities {
    const supported = this.projection.transport === 'api' || this.projection.runtime === 'grok-cli' || this.projection.runtime === 'pi'
    const platform = process.platform !== 'win32'
    const ready = supported && platform
    const cli = ready && this.projection.transport === 'agent-cli'
    return {
      supported: ready,
      ...(!supported ? { code: 'unsupported_runtime' as const } : !platform ? { code: 'platform_unsupported' as const } : {}),
      availability: 'unchecked',
      resume: cli,
      workingDirectory: ready,
      toolPolicies: ready ? ['read-only', 'standard'] : [],
      hostTools: cli,
      nativeCallId: cli && this.projection.runtime === 'pi',
      forwarding: cli ? ['serial', 'parallel'] : [],
      timeout: ready,
      cancel: ready,
    }
  }
  createContext(cwd: string, storage?: CliStorage): ExecutionContext {
    this.assertSupported()
    const contextId = randomUUID()
    const cli = this.projection.transport === 'agent-cli'
    if (!cli && storage) throw new ExecutionError('invalid_request')
    const dedicated = cli ? resolveCliStorage(this.projection.runtime, storage) : undefined
    const context: ExecutionContext = { version: 1, contextId, connection: { ...this.projection }, cwd: workingDirectory(cwd), hasExecuted: false,
      ...(dedicated ?? {}),
      ...(this.projection.runtime === 'grok-cli' ? { nativeSessionId: randomUUID() } : {}),
      ...(this.projection.runtime === 'pi' ? { nativeSessionFile: join(dedicated!.sessionDir, `${contextId}.jsonl`) } : {}) }
    this.store.write('contexts', contextId, context)
    return context
  }
  getContext(contextId: string): ExecutionContext { return this.store.context(contextId) }
  start(contextId: string, input: string, constraints: ExecutionConstraints = {}, handler?: ToolHandler): string {
    this.assertSupported()
    if (typeof input !== 'string' || !input.trim() || input.length > 1024 * 1024) throw new ExecutionError('invalid_request')
    if (!constraints || Array.isArray(constraints) || typeof constraints !== 'object') throw new ExecutionError('unsupported_constraint')
    const allowed = new Set(['toolPolicy', 'timeoutMs', 'tools', 'forwarding'])
    if (Object.keys(constraints).some(key => !allowed.has(key))) throw new ExecutionError('unsupported_constraint')
    const hasTools = constraints.tools !== undefined
    if (hasTools && constraints.toolPolicy !== undefined) throw new ExecutionError('unsupported_constraint')
    if (!hasTools && constraints.forwarding !== undefined) throw new ExecutionError('unsupported_constraint')
    const timeoutMs = constraints.timeoutMs ?? 120000
    if (!Number.isInteger(timeoutMs) || timeoutMs <= 0 || timeoutMs > 3600000) throw new ExecutionError('unsupported_constraint')
    const toolPolicy = constraints.toolPolicy ?? 'read-only'
    let hostTools: { tools: HostToolSpec[]; forwarding: 'serial' | 'parallel' } | undefined
    if (hasTools) {
      if (this.projection.transport !== 'agent-cli' || !handler) throw new ExecutionError(this.projection.transport === 'agent-cli' ? 'invalid_request' : 'unsupported_constraint')
      if (!Array.isArray(constraints.tools) || constraints.tools.length === 0) throw new ExecutionError('invalid_request')
      const forwarding = constraints.forwarding ?? 'serial'
      if (forwarding !== 'serial' && forwarding !== 'parallel') throw new ExecutionError('unsupported_constraint')
      let tools: HostToolSpec[]
      try { tools = structuredClone(constraints.tools) } catch { throw new ExecutionError('invalid_request') }
      assertHostTools(tools)
      hostTools = { tools, forwarding }
    } else if (toolPolicy !== 'read-only' && toolPolicy !== 'standard') throw new ExecutionError('unsupported_constraint')
    let context = this.store.context(contextId)
    if (JSON.stringify(context.connection) !== JSON.stringify(this.projection)) throw new ExecutionError('connection_mismatch')
    workingDirectory(context.cwd)
    const worker = join(__dirname, 'worker.js')
    if (!existsSync(worker)) throw new ExecutionError('process_failed')
    const runId = randomUUID()
    // Keep claim, startup and synchronous rollback under one lock. Otherwise a
    // contender can prevent cleanup before a recoverable run record exists.
    return this.store.exclusive(contextId, () => {
      if (this.pendingCalls(contextId).length > 0) throw new ExecutionError('context_busy')
      this.releaseSettledClaim(contextId)
      this.store.claim(contextId, runId)
      let child: ChildProcess | undefined
      try {
        // A call can be recorded after the first scan and before this claim.
        if (this.pendingCalls(contextId).length > 0) throw new ExecutionError('context_busy')
        // The prior run may have finalized between the first read and this claim.
        context = this.store.context(contextId)
        if (JSON.stringify(context.connection) !== JSON.stringify(this.projection)) throw new ExecutionError('connection_mismatch')
        if (context.connection.transport === 'agent-cli') {
          prepareCliStorage(context)
          assertNativeSession(context)
        }
        const record: ExecutionRecord = { version: 1, runId, contextId, nativeSessionId: context.nativeSessionId, status: 'starting', startedAt: Date.now(), heartbeatAt: Date.now(), stopped: false }
        this.store.write('runs', runId, record)
        const childEnv = context.connection.transport === 'agent-cli' ? supervisorEnvironment(this.env) : this.env
        child = fork(worker, [], { env: childEnv, detached: !hostTools, stdio: hostTools ? ['ignore', 'ignore', 'ignore', 'ipc', 'pipe'] : ['ignore', 'ignore', 'ignore', 'ipc'], execArgv: [] })
        const pipe = hostTools ? child.stdio[4] : undefined
        if (hostTools && (!pipe || typeof pipe === 'string' || !('on' in pipe))) throw new ExecutionError('process_failed')
        if (pipe && typeof pipe !== 'string') {
          const stream = pipe as NodeJS.EventEmitter & { unref?: () => void }
          stream.on('error', () => { /* Worker exit closes this end after the run is stored. */ })
          stream.unref?.()
          this.parentPipes.push(stream)
        }
        const fail = () => {
          // Once IPC was accepted execution may have started: don't release its claim.
          try {
            const current = this.store.run(runId)
            if (current && ACTIVE.has(current.status)) { current.status = 'unknown'; current.code = 'process_failed'; this.store.write('runs', runId, current) }
          } catch { /* A stale starting record remains unknown; never crash the host or release its claim. */ }
        }
        child.on('error', fail)
        if (hostTools && handler) {
          child.on('message', (message: WorkerToolMessage) => {
            if (!message || message.type !== 'tool-call' || !message.call) return
            void Promise.resolve().then(() => handler(message.call)).then(result => normalizeToolResult(result), () => normalizeToolResult(undefined)).then(result => {
              try { child?.send({ type: 'tool-result', callId: message.call.callId, result }) } catch { /* The worker has already exited. */ }
            })
          })
        }
        const message: WorkerRequest = { dataDir: this.store.root, context, record, connection: this.connection, input, constraints: { toolPolicy: hostTools ? undefined : toolPolicy, timeoutMs }, ...(hostTools ? { hostTools } : {}) }
        child.send(message, err => { if (err) fail() })
        if (!hostTools) child.unref()
        return runId
      } catch (e) {
        if (child && child.exitCode === null) child.kill('SIGKILL')
        this.store.release(contextId, runId); throw e
      }
    })
  }
  toolCall(callId: string): ToolCallRecord | undefined { return this.store.read<ToolCallRecord>('calls', callId) }
  /** Pending calls for a context, including ones persisted before the handler ran. */
  pendingToolCalls(contextId: string): ToolCallRecord[] {
    this.assertSupported()
    this.store.context(contextId)
    return this.pendingCalls(contextId)
  }
  /** Record the host's checked result for a call whose reply never reached the CLI. */
  reconcile(callId: string, output: string): ToolCallRecord {
    this.assertSupported()
    if (typeof output !== 'string' || output.length === 0 || output.length > 65536) throw new ExecutionError('invalid_request')
    const existing = this.toolCall(callId)
    if (!existing) throw new ExecutionError('invalid_request')
    return this.store.exclusive(existing.contextId, () => {
      const call = this.toolCall(callId)
      if (!call) throw new ExecutionError('invalid_request')
      if (call.status === 'reconciled') {
        if (call.output !== output) throw new ExecutionError('invalid_request')
        const saved = this.store.run(call.runId)
        if (!saved) throw new ExecutionError('invalid_request')
        if (ACTIVE.has(saved.status)) throw new ExecutionError('context_busy')
        this.releaseSettledClaim(call.contextId)
        return call
      }
      if (call.status !== 'pending') throw new ExecutionError('invalid_request')
      const run = this.store.run(call.runId)
      if (!run) throw new ExecutionError('invalid_request')
      if (ACTIVE.has(run.status)) throw new ExecutionError('context_busy')
      const updated: ToolCallRecord = { ...call, status: 'reconciled', output }
      this.store.write('calls', call.callId, updated)
      this.releaseSettledClaim(call.contextId)
      return updated
    })
  }
  /** Drop a claim whose run has stopped and whose calls are already reconciled. */
  private releaseSettledClaim(contextId: string): void {
    if (this.pendingCalls(contextId).length > 0) return
    const claim = this.store.path('active', contextId)
    if (!existsSync(claim)) return
    const runId = readFileSync(claim, 'utf8')
    const run = this.store.run(runId)
    if (!run || run.contextId !== contextId) throw new ExecutionError('storage_error')
    if (!run.stopped || ACTIVE.has(run.status)) return
    this.store.release(contextId, run.runId)
  }
  private pendingCalls(contextId: string): ToolCallRecord[] {
    const dir = join(this.store.root, 'calls')
    if (!existsSync(dir)) return []
    const found: ToolCallRecord[] = []
    for (const name of readdirSync(dir)) {
      if (!name.endsWith('.json')) continue
      const call = this.store.read<ToolCallRecord>('calls', name.slice(0, -'.json'.length))
      if (call && call.contextId === contextId && call.status === 'pending') found.push(call)
    }
    return found
  }
  query(runId: string): ExecutionRecord | undefined {
    const record = this.store.run(runId)
    if (record && ACTIVE.has(record.status) && Date.now() - record.heartbeatAt > 5000) return { ...record, status: 'unknown', stopped: false }
    return record
  }
  async wait(runId: string, timeoutMs = 130000): Promise<ExecutionRecord> {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new ExecutionError('invalid_request')
    const until = Date.now() + timeoutMs
    do {
      const record = this.query(runId)
      if (!record) throw new ExecutionError('invalid_request')
      if (!ACTIVE.has(record.status)) return record
      await delay(25)
    } while (Date.now() < until)
    return this.query(runId)!
  }
  async cancel(runId: string): Promise<ExecutionRecord> {
    const record = this.query(runId)
    if (!record) throw new ExecutionError('invalid_request')
    if (!ACTIVE.has(record.status) && record.status !== 'unknown') return record
    this.store.requestCancel(runId)
    const result = await this.wait(runId, 9500)
    return ACTIVE.has(result.status) ? { ...result, status: 'unknown', stopped: false } : result
  }
  private assertSupported(): void {
    const capability = this.capabilities()
    if (!capability.supported) throw new ExecutionError(capability.code!)
  }
}
