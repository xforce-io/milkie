import { spawn } from 'node:child_process'
import { writeFileSync, unlinkSync } from 'node:fs'
import { join } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { resolveAndParseConnection } from '../connection/parse.js'
import { assembleApiGateway } from '../connection/assemble.js'
import { CliEvents, assertNativeSession, classifyFailure, cliCommand, cliEnvironment, prepareCliStorage } from './adapters.js'
import { acquireGrokConfigLock, assertPiHostConfig, inspectGrok, openToolBridge, visibleHostTools, watchParentPipe, writeGrokHostConfig, writePiExtension, type ToolBridge } from './hostTools.js'
import { ProcessTracker, EXECUTION_TOKEN_ENV } from './processes.js'
import { ExecutionStore } from './store.js'
import { ExecutionError, type ToolCall, type WorkerRequest, type WorkerToolReply, type ExecutionStatus } from './types.js'

interface ParentLife { dead: boolean }

export async function runExecution(request: WorkerRequest, life?: ParentLife): Promise<void> {
  const store = new ExecutionStore(request.dataDir), { context, record, constraints } = request
  const hosted = request.hostTools
  const waiters = new Map<string, (result: unknown) => void>()
  if (hosted) {
    process.on('message', (message: WorkerToolReply) => {
      if (!message || message.type !== 'tool-result' || !message.callId) return
      const resolve = waiters.get(message.callId)
      if (!resolve) return
      waiters.delete(message.callId)
      resolve(message.result)
    })
  }
  let stopping: 'cancelled' | 'timed_out' | undefined
  let stop: (() => void) | undefined
  let terminal = false
  let cleanup: (() => Promise<boolean>) | undefined
  let promptPath: string | undefined
  let tracker: ProcessTracker | undefined
  let bridge: ToolBridge | undefined
  let releaseConfig: (() => void) | undefined
  const pulse = () => {
    if (terminal) return
    record.heartbeatAt = Date.now()
    if (tracker) record.resources = tracker.resources()
    store.write('runs', record.runId, record)
    if (life?.dead) { stop?.(); return }
    if (!stopping && (store.isCancelled(record.runId) || Date.now() - record.startedAt >= constraints.timeoutMs)) {
      stopping = store.isCancelled(record.runId) ? 'cancelled' : 'timed_out'
      stop?.()
    }
  }
  const timer = setInterval(() => { try { pulse() } catch { stopping = 'cancelled'; stop?.() } }, 50)
  const finish = (status: ExecutionStatus, stopped: boolean) => {
    if (terminal) return
    terminal = true; clearInterval(timer)
    if (tracker) record.resources = tracker.resources()
    record.status = status; record.stopped = stopped; record.finishedAt = Date.now(); record.heartbeatAt = Date.now()
    // Drop the claim before publishing the terminal record, so a visible stopped run is no longer held.
    if (status !== 'unknown') store.release(context.contextId, record.runId)
    store.write('runs', record.runId, record)
  }
  const onHost = (call: ToolCall) => new Promise<unknown>((resolve, reject) => {
    const timer = setInterval(() => { if (life?.dead || terminal) { clearInterval(timer); reject(new Error('Host unavailable.')) } }, 40)
    waiters.set(call.callId, result => { clearInterval(timer); resolve(result) })
    if (!process.send) { clearInterval(timer); reject(new Error('Host unavailable.')); return }
    process.send({ type: 'tool-call', call }, error => { if (error) { clearInterval(timer); waiters.delete(call.callId); reject(error) } })
  })
  try {
    pulse()
    if (life?.dead) { finish('unknown', true); return }
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
    let extensionPath: string | undefined
    let visibleTools = hosted?.tools
    if (hosted) {
      visibleTools = visibleHostTools(store, context.contextId, hosted.tools)
      const toolsFile = join(store.root, 'runs', `${record.runId}.tools.json`)
      writeFileSync(toolsFile, JSON.stringify(visibleTools), { mode: 0o600 })
      bridge = await openToolBridge({
        tools: hosted.tools, forwarding: hosted.forwarding, runId: record.runId, contextId: context.contextId, store,
        alive: () => !life?.dead && !terminal,
        onHost,
        onBroken: () => { record.code = 'process_failed'; stop?.() },
      })
      if (life?.dead) { finish('unknown', true); return }
      if (context.connection.runtime === 'grok-cli') {
        releaseConfig = acquireGrokConfigLock(context.configDir!, record.runId, previousRunId => store.run(previousRunId)?.stopped === true)
        const script = join(__dirname, 'mcp-server.js')
        const launchArgs = [script, bridge.socketPath, toolsFile]
        writeGrokHostConfig(context.configDir!, script, bridge.socketPath, toolsFile)
        await inspectGrok(context.cwd, cliEnvironment(process.env, context, { socketPath: bridge.socketPath, forwarding: hosted.forwarding }), join(context.configDir!, 'leader.sock'), { command: process.execPath, args: launchArgs }, () => !life?.dead)
      } else {
        assertPiHostConfig(context.configDir!)
        extensionPath = join(store.root, 'runs', `${record.runId}.extension.mjs`)
        writePiExtension(extensionPath, visibleTools, bridge.socketPath, hosted.forwarding)
      }
    }
    if (life?.dead) { finish('unknown', true); return }
    if (stopping) { finish(stopping, true); return }
    const promptFile = join(store.root, 'runs', `${record.runId}.prompt`)
    promptPath = promptFile
    const command = cliCommand(context, request.input, { toolPolicy: constraints.toolPolicy, timeoutMs: constraints.timeoutMs }, promptFile, hosted && visibleTools ? { names: visibleTools.map(tool => tool.name), extensionPath } : undefined)
    const childEnv = cliEnvironment(process.env, context, hosted && bridge ? { socketPath: bridge.socketPath, forwarding: hosted.forwarding } : undefined)
    // Grok takes a file; Pi takes stdin. Never expose a prompt in process argv.
    if (context.connection.runtime === 'grok-cli') writeFileSync(promptFile, request.input, { mode: 0o600, flag: 'wx' })
    const events = new CliEvents(context.connection.runtime as 'grok-cli' | 'pi')
    tracker = new ProcessTracker(record.runId)
    const child = spawn(command.command, command.args, { cwd: context.cwd, env: { ...childEnv, [EXECUTION_TOKEN_ENV]: record.runId }, detached: !hosted, stdio: ['pipe', 'pipe', 'pipe'] })
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
    if (child.pid) tracker.note(child.pid)
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
    const keepSession = () => {
      if (!events.sessionId || context.nativeSessionId) return
      context.nativeSessionId = events.sessionId
      record.nativeSessionId = events.sessionId
      store.write('contexts', context.contextId, context)
    }
    if (life?.dead) {
      keepSession()
      const stopped = outcome.kind === 'stop' ? outcome.stopped : await (cleanup?.() ?? Promise.resolve(true))
      finish('unknown', stopped)
      return
    }
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
    if (life?.dead) { finish('unknown', stopped); return }
    if (!stopped) { record.code = 'process_failed'; finish('unknown', false); return }
    if (stopping) { finish(stopping, true); return }
    if (record.code === 'process_failed' || processError || exitCode !== 0 || events.code || !events.ended || !events.sessionId) {
      record.code = record.code === 'process_failed' ? 'process_failed' : events.code ?? (exitCode === 0 ? 'protocol_error' : classifyFailure(stderr))
      finish('failed', true); return
    }
    assertNativeSession(context)
    record.output = events.output
    finish('succeeded', true)
  } catch (e) {
    if (life?.dead) {
      const stopped = cleanup ? await cleanup() : true
      finish('unknown', stopped)
      return
    }
    record.code = e instanceof ExecutionError ? e.code : classifyFailure(String(e instanceof Error ? e.message : ''))
    // API abort guarantees local request termination, not remote side-effect reversal.
    const stopped = cleanup ? await cleanup() : true
    finish(stopped ? (stopping ?? 'failed') : 'unknown', stopped)
  } finally {
    // Keep the config lock when stop was not confirmed. The next run must not rewrite it over a live descendant.
    if (record.stopped) releaseConfig?.()
    tracker?.close(); clearInterval(timer); await bridge?.close(); if (promptPath) { try { unlinkSync(promptPath) } catch { /* absent */ } }
  }
}

if (require.main === module) {
  const startup = setTimeout(() => process.exit(1), 10000)
  const life: ParentLife = { dead: false }
  process.once('message', (message: WorkerRequest) => {
    clearTimeout(startup)
    // The host holds the write end. EOF or IPC disconnect means it can no longer answer tool calls.
    const releaseParent = message.hostTools ? watchParentPipe(() => { life.dead = true }) : undefined
    if (message.hostTools) process.on('disconnect', () => { life.dead = true })
    else if (process.connected) process.disconnect()
    const exit = (code: number) => { releaseParent?.(); process.exit(code) }
    void runExecution(message, life).then(() => exit(0), () => exit(1))
  })
}
