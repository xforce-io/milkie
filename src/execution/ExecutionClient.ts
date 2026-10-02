import { fork } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { join } from 'node:path'
import { existsSync } from 'node:fs'
import { setTimeout as delay } from 'node:timers/promises'
import type { ConnectionInput } from '../connection/types.js'
import { resolveAndParseConnection } from '../connection/parse.js'
import { ExecutionStore, workingDirectory } from './store.js'
import { assertNativeSession, prepareCliStorage, resolveCliStorage, supervisorEnvironment } from './adapters.js'
import { ExecutionError, type CliStorage, type ExecutionCapabilities, type ExecutionClientOptions, type ExecutionConstraints, type ExecutionContext, type ExecutionRecord, type WorkerRequest } from './types.js'

const ACTIVE = new Set(['starting', 'running'])
export class ExecutionClient {
  private readonly store: ExecutionStore
  private readonly projection
  private readonly env: NodeJS.ProcessEnv
  private readonly connection: ConnectionInput
  constructor(options: ExecutionClientOptions) {
    this.connection = structuredClone(options.connection)
    this.projection = resolveAndParseConnection(this.connection).projection
    this.store = new ExecutionStore(options.dataDir)
    this.env = { ...(options.env ?? process.env) }
  }
  capabilities(): ExecutionCapabilities {
    const supported = this.projection.transport === 'api' || this.projection.runtime === 'grok-cli' || this.projection.runtime === 'pi'
    const platform = process.platform !== 'win32'
    return { supported: supported && platform, ...(!supported ? { code: 'unsupported_runtime' as const } : !platform ? { code: 'platform_unsupported' as const } : {}), availability: 'unchecked', resume: supported && platform && this.projection.transport === 'agent-cli', workingDirectory: supported && platform, toolPolicies: supported && platform ? ['read-only', 'standard'] : [], timeout: supported && platform, cancel: supported && platform }
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
  start(contextId: string, input: string, constraints: ExecutionConstraints = {}): string {
    this.assertSupported()
    if (typeof input !== 'string' || !input.trim() || input.length > 1024 * 1024) throw new ExecutionError('invalid_request')
    if (!constraints || Array.isArray(constraints) || typeof constraints !== 'object' || Object.keys(constraints).some(k => k !== 'toolPolicy' && k !== 'timeoutMs')) throw new ExecutionError('unsupported_constraint')
    const normalized: Required<ExecutionConstraints> = { toolPolicy: constraints.toolPolicy ?? 'read-only', timeoutMs: constraints.timeoutMs ?? 120000 }
    if (!['read-only', 'standard'].includes(normalized.toolPolicy) || !Number.isInteger(normalized.timeoutMs) || normalized.timeoutMs <= 0 || normalized.timeoutMs > 3600000) throw new ExecutionError('unsupported_constraint')
    let context = this.store.context(contextId)
    if (JSON.stringify(context.connection) !== JSON.stringify(this.projection)) throw new ExecutionError('connection_mismatch')
    workingDirectory(context.cwd)
    const worker = join(__dirname, 'worker.js')
    if (!existsSync(worker)) throw new ExecutionError('process_failed')
    const runId = randomUUID()
    this.store.claim(contextId, runId)
    try {
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
      const child = fork(worker, [], { env: childEnv, detached: true, stdio: ['ignore', 'ignore', 'ignore', 'ipc'], execArgv: [] })
      const fail = () => {
        // Once IPC was accepted execution may have started: don't release its claim.
        try {
          const current = this.store.run(runId)
          if (current && ACTIVE.has(current.status)) { current.status = 'unknown'; current.code = 'process_failed'; this.store.write('runs', runId, current) }
        } catch { /* A stale starting record remains unknown; never crash the host or release its claim. */ }
      }
      child.on('error', fail)
      const message: WorkerRequest = { dataDir: this.store.root, context, record, connection: this.connection, input, constraints: normalized }
      child.send(message, err => { if (err) fail() })
      child.unref()
      return runId
    } catch (e) { this.store.release(contextId, runId); throw e }
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
