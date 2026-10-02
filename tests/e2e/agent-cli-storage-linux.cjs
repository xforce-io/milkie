#!/usr/bin/env node
// Linux container evidence for #265 S1.A3 and S2.A5. Requires MILKIE_LINUX_STORAGE=1.
const { spawn } = require('node:child_process')
const { mkdtempSync, mkdirSync, copyFileSync, chmodSync, existsSync, readFileSync, writeFileSync, rmSync } = require('node:fs')
const { homedir, tmpdir } = require('node:os')
const { join, resolve } = require('node:path')
const { randomUUID } = require('node:crypto')
const assert = require('node:assert/strict')
const { ExecutionClient } = require('../../dist/execution/ExecutionClient')
const { nativeFile } = require('../../dist/execution/adapters')

function evidence(value) { console.log(JSON.stringify(value)) }
function prepare(root, runtime) {
  const configDir = join(root, 'config'), sessionDir = join(root, 'sessions')
  mkdirSync(configDir, { recursive: true, mode: 0o700 })
  mkdirSync(sessionDir, { recursive: true, mode: 0o700 })
  const source = runtime === 'grok-cli' ? '/auth/grok-auth.json' : '/auth/pi-auth.json'
  if (!existsSync(source)) throw new Error(`Login material is not prepared for ${runtime}.`)
  copyFileSync(source, join(configDir, 'auth.json'))
  chmodSync(join(configDir, 'auth.json'), 0o600)
  if (runtime === 'pi' && existsSync('/auth/pi-settings.json')) copyFileSync('/auth/pi-settings.json', join(configDir, 'settings.json'))
  return { configDir, sessionDir }
}
function turn(request) {
  const child = spawn(process.execPath, [resolve('tests/fixtures/execution-host.cjs')], { stdio: ['pipe', 'pipe', 'inherit'] })
  let output = ''
  child.stdout.on('data', chunk => { output += chunk })
  const exit = new Promise((resolveExit, reject) => { child.on('error', reject); child.on('exit', resolveExit) })
  child.stdin.end(JSON.stringify(request))
  return exit.then(code => {
    const events = output.trim().split('\n').filter(Boolean).map(line => JSON.parse(line))
    return { code, events }
  })
}
async function cancelRunning(runtime, client, connection, dataDir, cwd) {
  const task = join(cwd, 'task.cjs')
  writeFileSync(task, "const fs=require('fs'),cp=require('child_process');const name=process.argv[2];const child=cp.spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{detached:true,stdio:'ignore'});fs.writeFileSync(name+'.pid',String(child.pid));setInterval(()=>{},1000);\n")
  const input = `Use your terminal/bash tool to execute this exact command in the current directory: node ${JSON.stringify(task)} cancel-target. Run it now; it intentionally waits. Do not alter the program.`
  const host = spawn(process.execPath, [resolve('tests/fixtures/execution-host.cjs')], { stdio: ['pipe', 'pipe', 'inherit'] })
  let buffer = ''
  const started = new Promise((resolveStarted, reject) => {
    host.on('error', reject)
    host.stdout.on('data', chunk => {
      buffer += chunk
      const end = buffer.indexOf('\n')
      if (end >= 0) resolveStarted(JSON.parse(buffer.slice(0, end)))
    })
  })
  const context = client.createContext(cwd, prepare(join(cwd, 'cancel-storage'), runtime))
  host.stdin.end(JSON.stringify({ connection, dataDir, cwd, contextId: context.contextId, input, constraints: { toolPolicy: 'standard', timeoutMs: 120000 } }))
  const { runId } = await started
  const pidFile = join(cwd, 'cancel-target.pid')
  let pid
  for (let i = 0; i < 1200; i++) {
    if (existsSync(pidFile)) { pid = Number(readFileSync(pidFile, 'utf8')); break }
    const current = client.query(runId)
    if (current && !['starting', 'running'].includes(current.status)) throw new Error(`Cancel target exited early: ${JSON.stringify(current)}`)
    await new Promise(resolveWait => setTimeout(resolveWait, 50))
  }
  if (!pid) throw new Error('CLI did not start the task process.')
  const result = await client.cancel(runId)
  let alive = true
  try { process.kill(pid, 0) } catch { alive = false }
  evidence({ runtime, test: 'cancel', result, pid, alive })
  assert.equal(result.status, 'cancelled')
  assert.equal(result.stopped, true)
  assert.equal(alive, false)
  if (host.exitCode === null) host.kill('SIGKILL')
}

async function main() {
  if (process.env.MILKIE_LINUX_STORAGE !== '1') throw new Error('Explicit MILKIE_LINUX_STORAGE=1 is required.')
  const mounts = readFileSync('/proc/mounts', 'utf8')
  if (mounts.includes('/.grok/sessions') || mounts.includes('/.pi/agent/sessions')) throw new Error('Host session storage is mounted.')
  evidence({ type: 'environment', platform: process.platform, arch: process.arch, node: process.version })
  const homeCli = [join(homedir(), '.grok'), join(homedir(), '.pi')]
  const assertHomeClean = (when) => {
    for (const dir of homeCli) if (existsSync(dir)) throw new Error(`Container HOME CLI config appeared ${when}: ${dir}`)
  }
  assertHomeClean('before')
  const root = mkdtempSync(join(tmpdir(), 'milkie-265-linux-'))
  const only = process.env.MILKIE_LINUX_RUNTIMES
  for (const runtime of ['grok-cli', 'pi'].filter(runtime => !only || only.split(',').includes(runtime))) {
    const connection = { contractVersion: 1, fields: { transport: 'agent-cli', runtime } }
    const dataDir = join(root, `${runtime}-data`)
    const client = new ExecutionClient({ dataDir, connection })
    const base = join(root, runtime)
    const storage = prepare(base, runtime)
    const work = join(base, 'work')
    mkdirSync(work)
    assert.throws(() => client.createContext(work, { configDir: join(base, 'absent-config'), sessionDir: storage.sessionDir }), /config_missing/)
    evidence({ runtime, test: 'config-missing', result: 'pass' })
    const context = client.createContext(work, storage)
    const marker = `synthetic-${randomUUID()}`
    const first = await turn({ connection, dataDir, cwd: work, contextId: context.contextId, input: `Remember exactly this synthetic marker: ${marker}. Reply ACK only. Do not use tools.` })
    const firstResult = first.events.find(event => event.type === 'result')?.result
    evidence({ runtime, test: 'first-turn', exitCode: first.code, status: firstResult?.status, code: firstResult?.code })
    assert.equal(first.code, 0)
    assert.equal(firstResult.status, 'succeeded')
    const sessionFile = nativeFile(client.getContext(context.contextId))
    assert.equal(sessionFile.startsWith(client.getContext(context.contextId).sessionDir), true)
    const second = await turn({ connection, dataDir, cwd: work, contextId: context.contextId, input: 'Repeat only the synthetic marker previously given in this conversation. Do not use tools or read files.' })
    const secondResult = second.events.find(event => event.type === 'result').result
    assert.equal(secondResult.status, 'succeeded')
    assert.equal(secondResult.nativeSessionId, firstResult.nativeSessionId)
    assert.ok(secondResult.output.includes(marker))
    rmSync(sessionFile)
    assert.throws(() => client.start(context.contextId, 'Continue.'), /session_missing/)
    assert.equal(existsSync(sessionFile), false)
    evidence({ runtime, test: 'resume-and-missing-session', result: 'pass', nativeSessionId: firstResult.nativeSessionId })
    if (runtime === 'grok-cli') await cancelRunning(runtime, client, connection, dataDir, work)
  }
  assertHomeClean('after')
  evidence({ test: 'container-home-untouched', result: 'pass' })
  evidence({ result: 'pass' })
}
main().catch(error => { evidence({ result: 'fail', reason: error instanceof Error ? error.message : 'probe failed' }); process.exitCode = 1 })
