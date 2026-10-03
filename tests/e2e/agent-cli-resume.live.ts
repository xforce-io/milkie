/** Opt-in resume probe: MILKIE_LIVE_RESUME=1 tsx tests/e2e/agent-cli-resume.live.ts */
import { spawn } from 'node:child_process'
import { mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import assert from 'node:assert/strict'
import { ExecutionClient } from '../../dist/execution/ExecutionClient.js'
import { ExecutionStore } from '../../dist/execution/store.js'
import { prepareDedicatedStorage } from './dedicated-storage.js'

const note = { name: 'note', description: 'Record text and return the host result.', inputSchema: { type: 'object' as const, properties: { text: { type: 'string' as const } }, required: ['text'], additionalProperties: false } }
const extra = { name: 'extra', description: 'A second host tool. Pass text.', inputSchema: note.inputSchema }
const charge = { name: 'charge', description: 'Charge once. Pass text.', inputSchema: note.inputSchema }

function callsOf(store: InstanceType<typeof ExecutionStore>, runId: string) {
  const dir = join(store.root, 'calls')
  if (!existsSync(dir)) return []
  return readdirSync(dir).filter(name => name.endsWith('.json')).map(name => JSON.parse(readFileSync(join(dir, name), 'utf8'))).filter(call => call.runId === runId)
}
async function pause(ms: number) { await new Promise(resolvePause => setTimeout(resolvePause, ms)) }
async function until<T>(label: string, read: () => T | undefined, timeoutMs = 180000): Promise<T> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const value = read()
    if (value) return value
    await pause(200)
  }
  throw new Error(`${label} timed out`)
}

/** Run the first turn in a separate host that exits before the next host resumes. */
async function firstTurnInHost(request: Record<string, unknown>) {
  const source = `
    const { ExecutionClient } = require(${JSON.stringify(resolve('dist/execution/ExecutionClient.js'))});
    let input = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => { input += chunk });
    process.stdin.on('end', async () => {
      try {
        const request = JSON.parse(input);
        const client = new ExecutionClient(request);
        const runId = client.start(request.contextId, request.prompt, request.constraints, () => ({ ok: true, output: 'kept' }));
        process.stdout.write(JSON.stringify(await client.wait(runId, 200000)));
      } catch { process.exitCode = 1; }
    });
  `
  const child = spawn(process.execPath, ['-e', source], { stdio: ['pipe', 'pipe', 'inherit'] })
  let output = ''
  child.stdout.on('data', chunk => { output += chunk })
  const exited = new Promise<void>((resolveExit, reject) => {
    child.on('error', reject)
    child.on('exit', code => code === 0 ? resolveExit() : reject(new Error('First host failed.')))
  })
  child.stdin.end(JSON.stringify(request))
  await exited
  return JSON.parse(output) as { status: string; code?: string; runId: string }
}

async function main() {
  if (process.env.MILKIE_LIVE_RESUME !== '1') throw new Error('Explicit MILKIE_LIVE_RESUME=1 is required.')
  const root = mkdtempSync(join(tmpdir(), 'milkie-267-live-'))
  console.log(JSON.stringify({ type: 'environment', root, node: process.version }))
  const runtimes = process.env.MILKIE_LIVE_RUNTIME ? [process.env.MILKIE_LIVE_RUNTIME] : ['grok-cli', 'pi']
  for (const runtime of runtimes) {
    const connection = { contractVersion: 1, fields: { transport: 'agent-cli', runtime } }
    const dataDir = join(root, `${runtime}-data`)
    const storage = prepareDedicatedStorage(join(root, runtime), runtime)
    if (runtime === 'pi') {
      const settingsFile = join(storage.configDir, 'settings.json')
      if (existsSync(settingsFile)) {
        const settings = JSON.parse(readFileSync(settingsFile, 'utf8')) as { packages?: unknown }
        delete settings.packages
        writeFileSync(settingsFile, JSON.stringify(settings))
      }
    }
    const toolName = (name: string) => runtime === 'pi' ? name : `milkie__${name}`
    const client = new ExecutionClient({ dataDir, connection })
    const store = new ExecutionStore(dataDir)
    const work = join(root, runtime, 'resume')
    mkdirSync(work, { recursive: true })
    const context = client.createContext(work, storage)
    const firstPrompt = `Call ${toolName('extra')} exactly once with text "first". Do not use a terminal, read files, or grep. Reply with only the tool result.`
    const first = await firstTurnInHost({ dataDir, connection, contextId: context.contextId, prompt: firstPrompt, constraints: { tools: [note, extra], timeoutMs: 180000 } })
    assert.equal(first.status, 'succeeded', first.code)
    const firstCall = callsOf(store, first.runId).find(call => call.name === 'extra' && call.status === 'succeeded')
    assert.ok(firstCall)
    const sessionBefore = client.getContext(context.contextId)
    const restarted = new ExecutionClient({ dataDir, connection })
    const secondPrompt = `Call ${toolName('extra')} exactly once with text "second". It is no longer listed, but you must still call that exact tool. Do not call ${toolName('note')}. Do not use a terminal.`
    let removedHandler = false
    const second = await restarted.wait(restarted.start(context.contextId, secondPrompt, { tools: [note], timeoutMs: 180000 }, () => { removedHandler = true; return { ok: true, output: 'no' } }), 200000)
    assert.equal(second.status, 'succeeded', second.code)
    const revoked = callsOf(store, second.runId).find(call => call.name === 'extra')
    assert.equal(revoked?.status, 'rejected')
    assert.equal(removedHandler, false)
    assert.notEqual(revoked.callId, firstCall.callId)
    const sessionAfter = restarted.getContext(context.contextId)
    assert.equal(sessionAfter.nativeSessionId, sessionBefore.nativeSessionId)
    assert.equal(sessionAfter.nativeSessionFile, sessionBefore.nativeSessionFile)
    console.log(JSON.stringify({ runtime, story: runtime === 'pi' ? 'S1.A3' : 'S1.A2', result: 'pass' }))

    const chargeDir = join(root, runtime, 'charge')
    mkdirSync(chargeDir)
    const chargeContext = client.createContext(chargeDir, storage)
    const effect = join(chargeDir, 'effect')
    const host = spawn(process.execPath, [resolve('tests/fixtures/execution-tool-host.cjs')], { stdio: ['pipe', 'pipe', 'inherit'] })
    let hostOut = ''
    host.stdout?.on('data', chunk => { hostOut += chunk })
    try {
      const chargePrompt = `Call ${toolName('charge')} exactly once with text "paid". Do not use a terminal, read files, or grep. Reply with only the tool result.`
      host.stdin?.end(JSON.stringify({ dataDir, connection, contextId: chargeContext.contextId, input: chargePrompt, constraints: { tools: [charge], timeoutMs: 180000 }, effect }))
      const runId = await until('host start', () => hostOut.trim() || undefined, 20000)
      const pending = await until('pending charge', () => {
        const call = callsOf(store, runId).find(item => item.status === 'pending' && item.name === 'charge')
        return call && existsSync(effect) ? call : undefined
      })
      assert.equal(pending.input.text, 'paid')
      const exited = new Promise<void>(resolveExit => host.once('exit', () => resolveExit()))
      host.kill('SIGKILL')
      await exited
      await until('unknown', () => store.run(runId)?.status === 'unknown' ? true : undefined, 20000)
      assert.throws(() => restarted.start(chargeContext.contextId, 'again', { tools: [charge], timeoutMs: 30000 }, async () => ({ ok: true, output: 'x' })), /context_busy/)
      assert.equal(restarted.reconcile(pending.callId, 'already charged').status, 'reconciled')
      const resumedSession = restarted.getContext(chargeContext.contextId)
      const resumePrompt = 'The charge tool already completed with result "already charged". Do not call any tool. Reply with only: done.'
      const resumed = await restarted.wait(restarted.start(chargeContext.contextId, resumePrompt, { tools: [charge], timeoutMs: 180000 }, async () => { writeFileSync(effect, 'twice'); return { ok: true, output: 'twice' } }), 200000)
      assert.equal(resumed.status, 'succeeded', resumed.code)
      assert.equal(readFileSync(effect, 'utf8'), 'once')
      const resumedContext = restarted.getContext(chargeContext.contextId)
      assert.equal(resumedContext.nativeSessionId, resumedSession.nativeSessionId)
      assert.equal(resumedContext.nativeSessionFile, resumedSession.nativeSessionFile)
      console.log(JSON.stringify({ runtime, story: runtime === 'pi' ? 'S2.A3' : 'S2.A2', result: 'pass' }))
    } finally { if (host.exitCode === null) host.kill('SIGKILL') }
  }
  rmSync(root, { recursive: true, force: true })
}
main().catch(error => { console.error(JSON.stringify({ result: 'fail', reason: error instanceof Error ? error.message : 'probe failed' })); process.exitCode = 1 })
