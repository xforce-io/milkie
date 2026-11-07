import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync, chmodSync, existsSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
const { ExecutionClient } = require('../../dist/execution/ExecutionClient') as typeof import('../../src/execution/ExecutionClient')

// This suite exercises the distributed CLI and real worker processes with the
// existing deterministic CLI fixture. It is not native Harness acceptance.
let root: string
let children: ChildProcessWithoutNullStreams[]
const connection = { contractVersion: 1, fields: { transport: 'agent-cli', runtime: 'pi' } }
beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'milkie-sidecar-'))
  children = []
  mkdirSync(join(root, 'bin'))
  const fixture = resolve('tests/fixtures/agent-cli.cjs')
  chmodSync(fixture, 0o755)
  symlinkSync(fixture, join(root, 'bin', 'pi'))
  const codexFixture = resolve('tests/fixtures/codex-cli.cjs')
  chmodSync(codexFixture, 0o755)
  symlinkSync(codexFixture, join(root, 'bin', 'codex'))
  writeFileSync(join(root, 'connection.json'), JSON.stringify(connection))
})
afterEach(async () => {
  for (const c of children) {
    if (c.exitCode === null && c.signalCode === null) {
      c.stdin.end()
      await until(() => c.exitCode !== null || c.signalCode !== null, 15000)
    }
  }
  rmSync(root, { recursive: true, force: true })
})
async function until<T>(read: () => T, ms = 12000): Promise<NonNullable<T>> {
  const deadline = Date.now() + ms
  while (Date.now() < deadline) { const v = read(); if (v) return v as NonNullable<T>; await delay(20) }
  throw new Error('Timed out waiting for sidecar evidence.')
}
function sidecar(extraArgs?: string[], runtime = 'pi') {
  const selected = runtime === 'pi' ? connection : { contractVersion: 1, fields: { transport: 'agent-cli', runtime, model: 'test-model' } }
  writeFileSync(join(root, 'connection.json'), JSON.stringify(selected))
  const child = spawn(process.execPath, [resolve('dist/cli/index.js'), 'execution-sidecar', ...(extraArgs ?? ['--data-dir', join(root, 'data'), '--connection-file', join(root, 'connection.json')])], {
    env: { ...process.env, PATH: `${join(root, 'bin')}:${process.env.PATH}` },
  })
  children.push(child)
  const messages: any[] = []
  let stdout = '', stderr = '', buffer = '', serial = 0
  child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8')
  child.stdout.on('data', (s: string) => {
    stdout += s; buffer += s
    let end: number
    while ((end = buffer.indexOf('\n')) >= 0) { messages.push(JSON.parse(buffer.slice(0, end))); buffer = buffer.slice(end + 1) }
  })
  child.stderr.on('data', (s: string) => { stderr += s })
  const write = (m: unknown) => child.stdin.write(JSON.stringify(m) + '\n')
  const rpc = async (method: string, params: unknown = {}) => {
    const id = ++serial
    write({ jsonrpc: '2.0', id, method, params })
    return until(() => messages.find(m => m.id === id && !m.method))
  }
  const init = () => rpc('initialize', { protocolVersion: 1 })
  const context = async () => {
    const key = `ctx-${serial}`
    const cwd = join(root, key), configDir = join(root, `${key}-config`), sessionDir = join(root, `${key}-sessions`)
    for (const d of [cwd, configDir, sessionDir]) mkdirSync(d)
    writeFileSync(join(configDir, 'auth.json'), '{"fixture":true}\n')
    const r = await rpc('context.create', { cwd, storage: { configDir, sessionDir } })
    expect(r.error).toBeUndefined()
    return r.result as import('../../src/execution/types').ExecutionContext
  }
  const terminal = async (runId: string) => {
    const deadline = Date.now() + 15000
    while (Date.now() < deadline) {
      const r = await rpc('execution.query', { runId })
      if (r.result?.finishedAt) return r.result
      await delay(50)
    }
    throw new Error('Execution did not terminate.')
  }
  return { child, rpc, init, context, write, terminal, messages, stdout: () => stdout, stderr: () => stderr }
}
const alpha = { name: 'alpha', description: 'Alpha', inputSchema: { type: 'object', properties: { n: { type: 'integer' } }, required: ['n'], additionalProperties: false } }
const beta = { ...alpha, name: 'beta', description: 'Beta' }

test('version rejection is before execution; unsupported and unchecked capabilities are explicit', async () => {
  const s = sidecar()
  expect((await s.rpc('execution.start', { contextId: 'missing', input: 'must not start' })).error.code).toBe(-32002)
  expect((await s.rpc('initialize', { protocolVersion: 999 })).error.code).toBe(-32001)
  expect(existsSync(join(root, 'data', 'runs')) && require('node:fs').readdirSync(join(root, 'data', 'runs')).length > 0).toBe(false)
  expect((await s.init()).result).toMatchObject({ protocolVersion: 1, capabilities: { supported: true, availability: 'unchecked' } })
  expect((await s.rpc('execution.query', { runId: '0afbd3f5-461a-45b3-b011-919ca39efa72' })).result).toBeNull()
  expect((await s.rpc('initialize', { protocolVersion: 1 })).error.code).toBe(-32600)
  expect((await s.rpc('context.create', { cwd: root, storage: { configDir: root, sessionDir: root } })).error.data.code).toBe('invalid_request')
  expect(s.stderr()).toBe('')
})

test('two contexts remain separate; cancellation is responsive while the other execution completes', async () => {
  const s = sidecar(); await s.init()
  const a = await s.context(), b = await s.context()
  const x = (await s.rpc('execution.start', { contextId: a.contextId, input: 'fixture:sleep', constraints: { timeoutMs: 30000 } })).result.runId
  const y = (await s.rpc('execution.start', { contextId: b.contextId, input: 'independent-marker' })).result.runId
  expect(x).not.toBe(y)
  expect((await s.rpc('execution.cancel', { runId: x })).result).toMatchObject({ runId: x, contextId: a.contextId, status: 'cancelled', stopped: true })
  expect(await s.terminal(y)).toMatchObject({ runId: y, contextId: b.contextId, status: 'succeeded', output: 'independent-marker', stopped: true })
  expect(s.messages.some(m => m.method === 'execution.updated' && m.params.runId === x)).toBe(true)
  expect(s.messages.filter(m => m.method === 'execution.updated').every(m => !('output' in m.params))).toBe(true)
})

test('bidirectional tool results cannot cross calls; query/cancel can run while a reply is pending', async () => {
  const s = sidecar(); await s.init()
  const c = await s.context()
  const runId = (await s.rpc('execution.start', { contextId: c.contextId, input: 'fixture:tools', constraints: { tools: [alpha, beta], forwarding: 'parallel', timeoutMs: 15000 } })).result.runId
  const calls = await until(() => {
    const found = s.messages.filter(m => m.method === 'tool.call')
    return found.length === 2 ? found : undefined
  })
  expect(calls[0].id).not.toBe(calls[1].id)
  expect(calls.every((m: any) => m.params.runId === runId && m.params.contextId === c.contextId)).toBe(true)
  expect((await s.rpc('execution.query', { runId })).result.status).toBe('running')
  s.write({ jsonrpc: '2.0', id: 'tool:late-unknown', result: { ok: true, output: 'must-not-appear' } })
  for (const m of [...calls].reverse()) s.write({ jsonrpc: '2.0', id: m.id, result: { ok: true, output: `result-${m.params.name}` } })
  expect(await s.terminal(runId)).toMatchObject({ status: 'succeeded', stopped: true })
  for (const m of calls) expect((await s.rpc('tool.get', { callId: m.params.callId })).result).toMatchObject({ name: m.params.name, output: `result-${m.params.name}`, status: 'succeeded' })
})

test.each(['eof', 'kill', 'sigterm'] as const)('%s of a tool-free host stops its worker and native process', async mode => {
  const s = sidecar(); await s.init()
  const c = await s.context()
  const runId = (await s.rpc('execution.start', { contextId: c.contextId, input: 'fixture:sleep', constraints: { timeoutMs: 30000 } })).result.runId
  const pidFile = join(c.cwd, 'runner.pid')
  await until(() => existsSync(pidFile))
  const pid = Number(readFileSync(pidFile, 'utf8'))
  const observer = new ExecutionClient({ dataDir: join(root, 'data'), connection })
  if (mode === 'eof') s.child.stdin.end()
  else s.child.kill(mode === 'kill' ? 'SIGKILL' : 'SIGTERM')
  await until(() => s.child.exitCode !== null || s.child.signalCode !== null)
  const r = await until(() => { const r = observer.query(runId); return r?.finishedAt ? r : undefined }, 15000)
  expect(r.stopped).toBe(true)
  expect(mode !== 'kill' ? r.status === 'cancelled' : r.status === 'unknown').toBe(true)
  expect(() => process.kill(pid, 0)).toThrow()
})

test('invalid constraints fail before spawning, malformed/oversized frames fail explicitly, secrets stay out of diagnostics', async () => {
  const s = sidecar(); await s.init()
  const c = await s.context()
  expect((await s.rpc('execution.start', { contextId: c.contextId, input: 'private-prompt', constraints: { tokenLimit: 1 } })).error.data.code).toBe('unsupported_constraint')
  expect(existsSync(join(c.cwd, 'runner.pid'))).toBe(false)
  s.child.stdin.write('not-json\n')
  await until(() => s.messages.some(m => m.error?.code === -32700))
  s.child.stdin.write('x'.repeat(8 * 1024 * 1024 + 1))
  await until(() => s.child.exitCode !== null)
  expect(s.messages.some(m => m.error?.message === 'Frame too large.')).toBe(true)
  expect(s.stdout() + s.stderr()).not.toContain('private-prompt')
  const bad = sidecar(['--connection-file', '/private/secret', '--data-dir', root])
  await until(() => bad.child.exitCode !== null)
  expect(bad.child.exitCode).toBe(1)
  expect(bad.stdout()).toBe('')
  expect(bad.stderr()).toBe('{"error":{"code":"SIDECAR_START_FAILED"}}\n')
})

test('a lost host tool response stays pending and prevents automatic resume', async () => {
  const s = sidecar(); await s.init()
  const c = await s.context()
  const runId = (await s.rpc('execution.start', { contextId: c.contextId, input: 'fixture:hold', constraints: { tools: [alpha], timeoutMs: 30000 } })).result.runId
  const call = await until(() => s.messages.find(m => m.method === 'tool.call'))
  s.child.kill('SIGKILL')
  const observer = new ExecutionClient({ dataDir: join(root, 'data'), connection })
  await until(() => { const r = observer.query(runId); return r?.finishedAt && r.stopped }, 15000)
  expect(observer.pendingToolCalls(c.contextId)).toEqual([expect.objectContaining({ callId: call.params.callId, runId, status: 'pending' })])
  expect(() => observer.start(c.contextId, 'must-not-replay')).toThrow('context_busy')
})

test('concurrent duplicate IDs close the connection and never execute the second start', async () => {
  const s = sidecar(); await s.init()
  const c = await s.context()
  const first = { jsonrpc: '2.0', id: 'duplicate', method: 'execution.start', params: { contextId: c.contextId, input: 'fixture:sleep', constraints: { timeoutMs: 30000 } } }
  s.child.stdin.write(JSON.stringify(first) + '\n' + JSON.stringify({ ...first, params: { ...first.params, input: 'must-not-execute' } }) + '\n')
  await until(() => s.child.exitCode !== null)
  expect(s.messages.some(m => m.id === 'duplicate' && m.error?.message === 'Duplicate request id.')).toBe(true)
  const files = require('node:fs').readdirSync(join(root, 'data', 'runs')).filter((f: string) => /^[\da-f-]+\.json$/.test(f))
  expect(files).toHaveLength(1)
  expect(JSON.parse(readFileSync(join(root, 'data', 'runs', files[0]), 'utf8')).stopped).toBe(true)
})

test('a broken stdout channel stops its executions instead of leaving them detached', async () => {
  const s = sidecar(); await s.init()
  const c = await s.context()
  const runId = (await s.rpc('execution.start', { contextId: c.contextId, input: 'fixture:sleep', constraints: { timeoutMs: 30000 } })).result.runId
  await until(() => existsSync(join(c.cwd, 'runner.pid')))
  const pid = Number(readFileSync(join(c.cwd, 'runner.pid'), 'utf8'))
  s.child.stdout.destroy()
  await until(() => s.child.exitCode !== null)
  const observer = new ExecutionClient({ dataDir: join(root, 'data'), connection })
  expect(observer.query(runId)).toMatchObject({ status: 'cancelled', stopped: true })
  expect(() => process.kill(pid, 0)).toThrow()
})


test('Codex uses the same distributed sidecar entry for host tools and exact session resume', async () => {
  const s = sidecar(undefined, 'codex')
  expect((await s.init()).result.capabilities).toMatchObject({ supported: true, nativeCallId: true, modelIterations: false })
  const c = await s.context()
  const first = (await s.rpc('execution.start', { contextId: c.contextId, input: 'fixture:tools', constraints: { tools: [alpha, beta], forwarding: 'parallel' } })).result.runId
  const calls = await until(() => { const found = s.messages.filter(m => m.method === 'tool.call'); return found.length === 2 ? found : undefined })
  for (const call of calls) s.write({ jsonrpc: '2.0', id: call.id, result: { ok: true, output: `sidecar-${call.params.name}` } })
  const one = await s.terminal(first)
  expect(one).toMatchObject({ status: 'succeeded', stopped: true })
  expect(calls.map((m: any) => m.params.nativeCallId)).toEqual(['native-call-1', 'native-call-2'])
  const second = (await s.rpc('execution.start', { contextId: c.contextId, input: 'sidecar-codex-marker' })).result.runId
  expect(await s.terminal(second)).toMatchObject({ status: 'succeeded', output: 'sidecar-codex-marker', nativeSessionId: one.nativeSessionId })
  const third = (await s.rpc('execution.start', { contextId: c.contextId, input: 'fixture:recall' })).result.runId
  expect(await s.terminal(third)).toMatchObject({ status: 'succeeded', output: 'sidecar-codex-marker', nativeSessionId: one.nativeSessionId })
})

test.each(['pi', 'codex'])('%s observation notifications and incremental queries share stable IDs', async runtime => {
  const s = sidecar(undefined, runtime)
  const caps = (await s.init()).result.capabilities.observation
  expect(caps).toMatchObject({ version: 1, lifecycle: 'supervisor', hiddenReasoning: 'unavailable' })
  const c = await s.context()
  const observation = { traceparent: '00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01', maxEvents: 100, maxBytes: 65536 }
  const id = (await s.rpc('execution.start', { contextId: c.contextId, input: 'observation-sidecar-marker', constraints: { observation } })).result.runId
  expect(await s.terminal(id)).toMatchObject({ status: 'succeeded', stopped: true })
  const first = (await s.rpc('observation.read', { runId: id, limit: 2 })).result
  expect(first.events).toHaveLength(2)
  const all = (await s.rpc('observation.read', { runId: id, limit: 100 })).result
  expect(all.status).toMatchObject({ complete: true, state: 'finished' })
  expect(all.events.slice(0, 2)).toEqual(first.events)
  const remaining = (await s.rpc('observation.read', { runId: id, afterSequence: first.nextSequence })).result
  expect(remaining.events).toEqual(all.events.slice(2))
  await until(() => s.messages.some(m => m.method === 'execution.observation' && m.params.kind === 'execution.finished'))
  const pushed = s.messages.filter(m => m.method === 'execution.observation' && m.params.runId === id).map(m => m.params)
  expect(pushed).toEqual(all.events)
  expect(new Set(pushed.map(m => m.id)).size).toBe(pushed.length)
  expect(JSON.stringify(pushed)).not.toContain('observation-sidecar-marker')
  expect((await s.rpc('observation.read', { runId: id, limit: 0 })).error.data.code).toBe('invalid_request')
})

test('public observation.purge rejects active execution and remains idempotent across sidecar restart', async () => {
  const s = sidecar(); expect((await s.init()).result.capabilities.observation.purge).toBe(true)
  const context = await s.context()
  const started = await s.rpc('execution.start', { contextId: context.contextId, input: 'fixture:sleep', constraints: { timeoutMs: 30000, observation: { traceparent: '00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01', maxEvents: 100, maxBytes: 65536 } } })
  const runId = started.result.runId
  expect((await s.rpc('observation.purge', { runId })).error.data.code).toBe('context_busy')
  expect((await s.rpc('execution.cancel', { runId })).result.stopped).toBe(true)
  const clean = await s.rpc('observation.purge', { runId })
  expect(clean.result).toMatchObject({ state: 'purged', runId, contextId: context.contextId })
  expect((await s.rpc('observation.read', { runId })).result).toMatchObject({ events: [], status: { fault: 'expired', complete: false } })
  s.child.stdin.end(); await until(() => s.child.exitCode !== null || s.child.signalCode !== null)
  const again = sidecar(); await again.init()
  expect((await again.rpc('observation.purge', { runId })).result).toEqual(clean.result)
  expect((await again.rpc('execution.query', { runId })).result).toMatchObject({ status: 'cancelled', stopped: true })
})
