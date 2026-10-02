/** Opt-in real storage probe: MILKIE_LIVE_STORAGE=1 tsx tests/e2e/agent-cli-storage.live.ts */
import { spawn } from 'node:child_process'
import { mkdtempSync, existsSync, mkdirSync, rmSync, readdirSync, statSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import assert from 'node:assert/strict'
import { ExecutionClient } from '../../dist/execution/ExecutionClient'
import { nativeFile } from '../../dist/execution/adapters'
import { prepareDedicatedStorage } from './dedicated-storage'

function snapshot(dir: string): string[] {
  if (!existsSync(dir)) return []
  const found: string[] = []
  const walk = (current: string) => {
    for (const name of readdirSync(current)) {
      const path = join(current, name)
      let info
      try { info = statSync(path) } catch { continue }
      if (info.isDirectory()) walk(path)
      else found.push(path)
    }
  }
  walk(dir)
  return found.sort()
}

async function main() {
  if (process.env.MILKIE_LIVE_STORAGE !== '1') throw new Error('Explicit MILKIE_LIVE_STORAGE=1 is required.')
  const root = mkdtempSync(join(tmpdir(), 'milkie-265-live-'))
  const hostSessions = {
    'grok-cli': snapshot(join(homedir(), '.grok', 'sessions')),
    pi: snapshot(join(homedir(), '.pi', 'agent', 'sessions')),
  }
  const leaderBefore = existsSync(join(homedir(), '.grok', 'leader.sock')) ? statSync(join(homedir(), '.grok', 'leader.sock')).mtimeMs : 0
  console.log(JSON.stringify({ type: 'environment', root, node: process.version }))
  for (const runtime of (process.env.MILKIE_LIVE_RUNTIME ? [process.env.MILKIE_LIVE_RUNTIME] : ['grok-cli', 'pi'])) {
    const connection = { contractVersion: 1, fields: { transport: 'agent-cli', runtime } }
    const dataDir = join(root, `${runtime}-data`)
    const client = new ExecutionClient({ dataDir, connection })
    const base = join(root, runtime)
    const storage = prepareDedicatedStorage(base, runtime)
    assert.throws(() => client.createContext(join(base, 'missing-cwd')), /invalid_request|config_missing|ENOENT/)
    const missingConfig = client
    assert.throws(() => missingConfig.createContext(join(base, 'work'), { configDir: join(base, 'absent-config'), sessionDir: storage.sessionDir }), /config_missing/)
    assert.throws(() => client.createContext(join(base, 'work'), { configDir: storage.configDir, sessionDir: join(base, 'absent-sessions') }), /session_missing/)
    const work = join(base, 'work')
    mkdirSync(work)
    const context = client.createContext(work, storage)
    const marker = `synthetic-${randomUUID()}`
    const firstHost = spawn(process.execPath, [resolve('tests/fixtures/execution-host.cjs')], { stdio: ['pipe', 'pipe', 'inherit'] })
    let firstOut = ''
    firstHost.stdout.on('data', chunk => { firstOut += chunk })
    const firstExit = new Promise<number | null>((resolveExit, reject) => { firstHost.on('error', reject); firstHost.on('exit', resolveExit) })
    firstHost.stdin.end(JSON.stringify({ connection, dataDir, cwd: work, contextId: context.contextId, input: `Remember exactly this synthetic marker: ${marker}. Reply ACK only. Do not use tools.` }))
    assert.equal(await firstExit, 0)
    const first = JSON.parse(firstOut.trim().split('\n').pop()!).result
    assert.equal(first.status, 'succeeded')
    const sessionFile = nativeFile(client.getContext(context.contextId))
    assert.equal(sessionFile.startsWith(client.getContext(context.contextId).sessionDir!), true)
    assert.equal(existsSync(sessionFile), true)
    const secondHost = spawn(process.execPath, [resolve('tests/fixtures/execution-host.cjs')], { stdio: ['pipe', 'pipe', 'inherit'] })
    let secondOut = ''
    secondHost.stdout.on('data', chunk => { secondOut += chunk })
    const secondExit = new Promise<number | null>((resolveExit, reject) => { secondHost.on('error', reject); secondHost.on('exit', resolveExit) })
    secondHost.stdin.end(JSON.stringify({ connection, dataDir, cwd: work, contextId: context.contextId, input: 'Repeat only the synthetic marker previously given in this conversation. Do not use tools or read files.' }))
    assert.equal(await secondExit, 0)
    const second = JSON.parse(secondOut.trim().split('\n').pop()!).result
    assert.equal(second.status, 'succeeded')
    assert.equal(second.nativeSessionId, first.nativeSessionId)
    assert.notEqual(second.runId, first.runId)
    assert.ok(second.output.includes(marker))
    rmSync(sessionFile)
    assert.throws(() => client.start(context.contextId, 'Continue.'), /session_missing/)
    assert.equal(existsSync(sessionFile), false)
    const hostDir = runtime === 'grok-cli' ? hostSessions['grok-cli'] : hostSessions.pi
    const after = snapshot(runtime === 'grok-cli' ? join(homedir(), '.grok', 'sessions') : join(homedir(), '.pi', 'agent', 'sessions'))
    assert.deepEqual(after, hostDir)
    if (runtime === 'grok-cli') {
      const leader = join(homedir(), '.grok', 'leader.sock')
      const leaderAfter = existsSync(leader) ? statSync(leader).mtimeMs : 0
      assert.equal(leaderAfter, leaderBefore)
    }
    console.log(JSON.stringify({ runtime, result: 'pass', nativeSessionId: first.nativeSessionId, sessionFile }))
  }
}
main().catch(error => { console.error(JSON.stringify({ result: 'fail', reason: error instanceof Error ? error.message : 'probe failed' })); process.exitCode = 1 })
