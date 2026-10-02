import { spawn } from 'node:child_process'
import { writeFileSync, unlinkSync } from 'node:fs'
import { join } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { resolveAndParseConnection } from '../connection/parse.js'
import { assembleApiGateway } from '../connection/assemble.js'
import { CliEvents, assertNativeSession, classifyFailure, cliCommand, cliEnvironment, prepareCliStorage } from './adapters.js'
import { ProcessTracker, EXECUTION_TOKEN_ENV } from './processes.js'
import { ExecutionStore } from './store.js'
import { ExecutionError, type WorkerRequest, type ExecutionStatus } from './types.js'

export async function runExecution(request: WorkerRequest): Promise<void> {
  const store = new ExecutionStore(request.dataDir), { context, record, constraints } = request
  let stopping: 'cancelled' | 'timed_out' | undefined
  let stop: (() => void) | undefined
  let terminal = false
  let cleanup: (() => Promise<boolean>) | undefined
  let promptPath: string | undefined
  let tracker: ProcessTracker | undefined
  const pulse = () => {
    if (terminal) return
    record.heartbeatAt = Date.now()
    store.write('runs', record.runId, record)
    if (!stopping && (store.isCancelled(record.runId) || Date.now() - record.startedAt >= constraints.timeoutMs)) {
      stopping = store.isCancelled(record.runId) ? 'cancelled' : 'timed_out'
      stop?.()
    }
  }
  const timer = setInterval(() => { try { pulse() } catch { stopping = 'cancelled'; stop?.() } }, 50)
  const finish = (status: ExecutionStatus, stopped: boolean) => {
    terminal = true; clearInterval(timer)
    if (tracker) record.resources = tracker.resources()
    record.status = status; record.stopped = stopped; record.finishedAt = Date.now(); record.heartbeatAt = Date.now()
    store.write('runs', record.runId, record)
    if (status !== 'unknown') store.release(context.contextId, record.runId)
  }
  try {
    pulse()
    if (stopping) { finish(stopping, true); return }
    record.status = 'running'
    store.write('runs', record.runId, record)
    if (context.connection.transport === 'api') {
      const controller = new AbortController()
      stop = () => controller.abort()
      const { gateway } = assembleApiGateway(resolveAndParseConnection(request.connection))
      const response = await gateway.complete({ model: context.connection.model!, messages: [{ role: 'user', content: [{ type: 'text', text: request.input }] }] }, { signal: controller.signal })
      pulse() // Recheck the wall clock even when gateway work delayed the timer.
      record.output = response.content.filter(c => c.type === 'text').map(c => c.type === 'text' ? c.text : '').join('')
      finish(stopping ?? 'succeeded', true)
      return
    }
    prepareCliStorage(context)
    assertNativeSession(context)
    const promptFile = join(store.root, 'runs', `${record.runId}.prompt`)
    promptPath = promptFile
    const command = cliCommand(context, request.input, constraints, promptFile)
    const childEnv = cliEnvironment(process.env, context)
    // Grok takes a file; Pi takes stdin. Never expose a prompt in process argv.
    if (context.connection.runtime === 'grok-cli') writeFileSync(promptFile, request.input, { mode: 0o600, flag: 'wx' })
    const events = new CliEvents(context.connection.runtime as 'grok-cli' | 'pi')
    tracker = new ProcessTracker(record.runId)
    const child = spawn(command.command, command.args, { cwd: context.cwd, env: { ...childEnv, [EXECUTION_TOKEN_ENV]: record.runId }, detached: true, stdio: ['pipe', 'pipe', 'pipe'] })
    let stderr = '', processError = false, stopResult: Promise<boolean> | undefined
    let notifyStop: (stopped: boolean) => void
    const stopFinished = new Promise<{ kind: 'stop'; stopped: boolean }>(resolve => {
      notifyStop = stopped => resolve({ kind: 'stop', stopped })
    })
    stop = () => {
      if (child.pid && !stopResult) {
        stopResult = tracker!.stop()
        void stopResult.then(stopped => notifyStop(stopped))
      }
    }
    cleanup = async () => child.pid ? await (stopResult ?? tracker!.stop()) : true
    tracker.start()
    child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8')
    child.stdout.on('data', (chunk: string) => { events.push(chunk); if (events.code === 'protocol_error') stop?.() })
    child.stderr.on('data', (chunk: string) => { stderr = (stderr + chunk).slice(-16384) })
    child.stdin.on('error', () => { /* early process exit is classified below */ })
    const closed = new Promise<number | null>(resolve => {
      child.on('error', () => { processError = true; resolve(null) })
      child.on('exit', (code) => resolve(code))
    })
    context.hasExecuted = true
    store.write('contexts', context.contextId, context)
    child.stdin.end(command.stdin)
    const outcome = await Promise.race([closed.then(code => ({ kind: 'exit' as const, code })), stopFinished])
    if (outcome.kind === 'stop' && !outcome.stopped) {
      record.code = 'process_failed'
      finish('unknown', false)
      return
    }
    const exitCode = outcome.kind === 'exit' ? outcome.code : await closed
    // Draining stdout is bounded even if a task descendant inherited the pipe.
    const drained = new Promise<void>(resolve => child.once('close', () => resolve()))
    const stopped = await cleanup()
    await Promise.race([drained, delay(100)])
    events.finish()
    try { unlinkSync(promptFile) } catch { /* Pi has no prompt file */ }
    if (events.sessionId) {
      if (context.nativeSessionId && events.sessionId !== context.nativeSessionId) events.code = 'session_mismatch'
      else { context.nativeSessionId = events.sessionId; record.nativeSessionId = events.sessionId; store.write('contexts', context.contextId, context) }
    }
    pulse()
    if (!stopped) { record.code = 'process_failed'; finish('unknown', false); return }
    if (stopping) { finish(stopping, true); return }
    if (processError || exitCode !== 0 || events.code || !events.ended || !events.sessionId) {
      record.code = events.code ?? (exitCode === 0 ? 'protocol_error' : classifyFailure(stderr))
      finish('failed', true); return
    }
    assertNativeSession(context)
    record.output = events.output
    finish('succeeded', true)
  } catch (e) {
    record.code = e instanceof ExecutionError ? e.code : classifyFailure(String(e instanceof Error ? e.message : ''))
    // API abort guarantees local request termination, not remote side-effect reversal.
    const stopped = cleanup ? await cleanup() : true
    finish(stopped ? (stopping ?? 'failed') : 'unknown', stopped)
  } finally { tracker?.close(); clearInterval(timer); if (promptPath) { try { unlinkSync(promptPath) } catch { /* absent */ } } }
}

if (require.main === module) {
  const startup = setTimeout(() => process.exit(1), 10000)
  process.once('message', (message: WorkerRequest) => {
    clearTimeout(startup)
    // The worker owns cancellation after the initiating host disconnects.
    if (process.connected) process.disconnect()
    void runExecution(message).then(() => process.exit(0), () => process.exit(1))
  })
}
