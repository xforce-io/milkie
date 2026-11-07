import { mkdtempSync, mkdirSync, symlinkSync, chmodSync, writeFileSync, readFileSync, rmSync, appendFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import { setTimeout as delay } from 'node:timers/promises'
import { ObservationJournal, assertObservation, contentSummary, readObservations, reportedUsage } from '../execution/observation'
import { CliEvents } from '../execution/adapters'
import { createToolDispatcher } from '../execution/hostTools'
import { ExecutionStore } from '../execution/store'
import type { ExecutionRecord } from '../execution/types'
const { ExecutionClient } = require('../../dist/execution/ExecutionClient') as typeof import('../execution/ExecutionClient')
let root: string
const parent = '00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01'
const opts = { traceparent: parent, maxEvents: 100, maxBytes: 65536 }
const alpha = { name: 'alpha', description: 'Alpha', inputSchema: { type: 'object' as const, properties: { n: { type: 'integer' as const } }, required: ['n'], additionalProperties: false } }
const beta = { ...alpha, name: 'beta' }
beforeEach(() => { root = mkdtempSync(join(tmpdir(), 'milkie-observation-')) })
afterEach(() => rmSync(root, { recursive: true, force: true }))
function journal(options = opts) {
  const store = new ExecutionStore(join(root, 'data'))
  const run: ExecutionRecord = { version: 1, runId: randomUUID(), contextId: randomUUID(), startedAt: Date.now(), heartbeatAt: Date.now(), status: 'running', stopped: false, observation: options }
  const writer = new ObservationJournal(store, run, options)
  return { store, run, writer }
}
function cli(runtime: string) {
  const bin = join(root, 'bin'); mkdirSync(bin)
  const fixture = resolve(runtime === 'codex' ? 'tests/fixtures/codex-cli.cjs' : 'tests/fixtures/agent-cli.cjs'); chmodSync(fixture, 0o755)
  symlinkSync(fixture, join(bin, runtime === 'grok-cli' ? 'grok' : runtime))
  const connection = { contractVersion: 1, fields: { transport: 'agent-cli', runtime, model: 'test-model' } }
  const a = new ExecutionClient({ dataDir: join(root, 'data'), connection, env: { ...process.env, PATH: `${bin}:${process.env.PATH}` }, lifetime: 'host' })
  const cwd = join(root, 'cwd'), configDir = join(root, 'config'), sessionDir = join(root, 'sessions')
  for (const d of [cwd, configDir, sessionDir]) mkdirSync(d)
  writeFileSync(join(configDir, 'auth.json'), '{"fixture":true}\n')
  const c = a.createContext(cwd, { configDir, sessionDir })
  return { a, c }
}
async function terminal(a: ReturnType<typeof cli>['a'], runId: string) {
  const deadline = Date.now() + 15000
  while (Date.now() < deadline) { const r = a.query(runId); if (r?.finishedAt) return r; await delay(30) }
  await a.cancel(runId)
  throw new Error('Execution did not finish.')
}

test.each([
  { ...opts, traceparent: parent.toUpperCase() },
  { ...opts, traceparent: '00-' + '0'.repeat(32) + '-b7ad6b7169203331-01' },
  { ...opts, traceparent: '00-0af7651916cd43dd8448eb211c80319c-' + '0'.repeat(16) + '-01' },
  { ...opts, maxEvents: 1 }, { ...opts, maxBytes: 4095 }, { ...opts, maxBytes: NaN },
])('invalid explicit observation options reject: %j', bad => expect(() => assertObservation(bad)).toThrow('unsupported_constraint'))

test('pages replay stable IDs, source order and parent context; terminal is distinct from business success', () => {
  const { store, run, writer } = journal()
  for (let n = 0; n < 7; n++) writer.emit('agent.activity', { category: 'message' })
  writer.finish('failed', true, 'auth_failed')
  const page = readObservations(store, run, 0, 3), same = readObservations(store, run, 0, 3)
  expect(page.events).toEqual(same.events); expect(page.nextSequence).toBe(3)
  expect(page.status).toMatchObject({ complete: true, state: 'finished', lastSequence: 8 })
  expect(page.events.every(e => e.traceId === parent.slice(3, 35) && e.parentSpanId === parent.slice(36, 52))).toBe(true)
  const rest = readObservations(store, run, page.nextSequence, 100)
  expect(rest.events.map(e => e.sequence)).toEqual([4, 5, 6, 7, 8])
  expect(rest.events.at(-1)?.data).toMatchObject({ status: 'failed', code: 'auth_failed' })
  expect(new Set([...page.events, ...rest.events].map(e => e.id)).size).toBe(8)
})

test('bounded source reserves its terminal and explicitly reports dropped events', () => {
  const { store, run, writer } = journal({ ...opts, maxEvents: 2, maxBytes: 4096 })
  for (let n = 0; n < 8; n++) writer.emit('agent.activity', { category: 'message' })
  writer.finish('succeeded', true)
  const page = readObservations(store, run)
  expect(page.events).toHaveLength(2)
  expect(page.events.at(-1)?.kind).toBe('execution.finished')
  expect(page.status).toMatchObject({ state: 'incomplete', complete: false, fault: 'capacity', dropped: 7, lastSequence: 9 })
})

test('unfinished, disabled, corrupt tails and missing storage never report complete', () => {
  const { store, run, writer } = journal()
  writer.emit('agent.activity', { category: 'turn' })
  expect(readObservations(store, run).status.complete).toBe(false)
  expect(readObservations(store, { ...run, observation: undefined })).toMatchObject({ enabled: false, status: { state: 'disabled', complete: false } })
  writer.finish('succeeded', true)
  appendFileSync(join(store.root, 'observations', `${run.runId}.jsonl`), '{bad-tail')
  expect(readObservations(store, run).status).toMatchObject({ state: 'incomplete', complete: false, fault: 'corrupt' })
  rmSync(join(store.root, 'observations'), { recursive: true })
  expect(() => writer.emit('agent.activity', { category: 'message' })).not.toThrow()
  expect(readObservations(store, { ...run, finishedAt: Date.now() }).status).toMatchObject({ state: 'incomplete', complete: false, fault: 'unavailable' })
})

test('source loss during execution stays explicit and producer instrumentation does not throw', () => {
  const { store, run, writer } = journal()
  rmSync(join(store.root, 'observations'), { recursive: true })
  expect(() => { writer.emit('agent.activity', { category: 'message' }); writer.finish('succeeded', true) }).not.toThrow()
  expect(readObservations(store, { ...run, status: 'succeeded', finishedAt: Date.now() }).status.complete).toBe(false)
})

test('unknown or invalid usage is omitted, never synthesized as zero', () => {
  expect(reportedUsage(undefined, 'message', 'delta', { inputTokens: 'input' })).toBeUndefined()
  expect(reportedUsage({ input: NaN, output: -1 }, 'message', 'delta', { inputTokens: 'input', outputTokens: 'output' })).toBeUndefined()
  expect(reportedUsage({ input: 0 }, 'message', 'delta', { inputTokens: 'input', outputTokens: 'output' })).toEqual({ scope: 'message', aggregation: 'delta', inputTokens: 0 })
})

test('tool result and transport delivery are separate; pending survives and hashes contain no raw content', async () => {
  const { store, run, writer } = journal()
  const dispatch = createToolDispatcher({ tools: [alpha], forwarding: 'serial', runId: run.runId, contextId: run.contextId, store, observation: writer, alive: () => true, onHost: async call => {
    expect(call.traceparent?.slice(3, 35)).toBe(parent.slice(3, 35))
    return { ok: true, output: 'PRIVATE-RESULT' }
  } })
  await dispatch({ id: 'req', name: 'alpha', input: { n: 1 }, nativeCallId: 'native-1' }, 64, async () => false)
  writer.finish('unknown', true)
  const page = readObservations(store, run)
  expect(page.events.map(e => e.kind)).toEqual(['tool.requested', 'tool.host.started', 'tool.host.responded', 'tool.delivery', 'execution.finished'])
  const request = page.events[0]!, response = page.events[2]!
  expect(request.spanId).toBe(response.spanId); expect(request.parentSpanId).not.toBe(parent.slice(36, 52))
  expect(response.data).toMatchObject({ output: contentSummary('PRIVATE-RESULT'), outcome: 'succeeded' })
  expect(page.events[3]?.data).toMatchObject({ outcome: 'unknown' })
  expect(JSON.stringify(page)).not.toContain('PRIVATE-RESULT')
  const callId = (request.data as { callId: string }).callId
  expect(store.read('calls', callId)).toMatchObject({ status: 'pending' })
})

test('an interrupted in-flight call has unknown delivery; duplicate native IDs cannot repeat effects', async () => {
  const { store, run, writer } = journal()
  let effects = 0, release!: () => void
  const gate = new Promise<void>(r => { release = r })
  const dispatch = createToolDispatcher({ tools: [alpha], forwarding: 'parallel', runId: run.runId, contextId: run.contextId, store, observation: writer, alive: () => !run.finishedAt, onHost: async () => { effects++; await gate; return { ok: true, output: 'effect' } } })
  const message = { id: 'req', name: 'alpha', input: { n: 1 }, nativeCallId: 'stable-native' }
  const pending = dispatch(message, 64, async () => true)
  expect(() => dispatch(message, 64, async () => true)).toThrow('protocol_error')
  run.finishedAt = Date.now(); writer.finish('cancelled', true, undefined, 'cancelled'); release(); await pending
  expect(effects).toBe(1)
  const events = readObservations(store, run).events
  expect(events.filter(e => e.kind === 'tool.interrupted')).toEqual([expect.objectContaining({ data: expect.objectContaining({ reason: 'cancelled' }) })])
  expect(events.find(e => e.kind === 'tool.delivery')?.data).toMatchObject({ outcome: 'unknown' })
})

test.each(['pi', 'grok-cli', 'codex'])('%s public execution source records complete tool correlation without payloads', async runtime => {
  const { a, c } = cli(runtime), callbacks: string[] = []
  const runId = a.start(c.contextId, 'fixture:tools', { tools: [alpha, beta], forwarding: 'parallel', observation: opts }, call => {
    callbacks.push(call.callId); expect(call.traceparent).toMatch(/^00-0af7651916cd43dd8448eb211c80319c-/)
    return { ok: true, output: 'PRIVATE-RESULT' }
  })
  expect((await terminal(a, runId)).status).toBe('succeeded')
  const page = a.observations(runId)
  expect(page.status).toMatchObject({ complete: true, state: 'finished' })
  expect(page.events[0]?.kind).toBe('execution.started'); expect(page.events.at(-1)?.kind).toBe('execution.finished')
  expect(page.events.filter(e => e.kind === 'tool.requested')).toHaveLength(2)
  expect(page.events.filter(e => e.kind === 'tool.host.responded')).toHaveLength(2)
  expect(page.events.filter(e => e.kind === 'tool.delivery')).toEqual([expect.objectContaining({ data: expect.objectContaining({ outcome: 'delivered' }) }), expect.objectContaining({ data: expect.objectContaining({ outcome: 'delivered' }) })])
  expect(JSON.stringify(page)).not.toContain('PRIVATE-RESULT')
  for (const callId of callbacks) expect(a.toolCall(callId)).toMatchObject({ status: 'succeeded', traceparent: expect.any(String) })
})

test('execution failure, parameter rejection and permission rejection remain distinct', async () => {
  const { a, c } = cli('pi')
  const failed = a.start(c.contextId, 'fixture:reject', { tools: [alpha], observation: opts }, () => { throw new Error('PRIVATE-EXCEPTION') })
  expect((await terminal(a, failed)).status).toBe('succeeded')
  const page = a.observations(failed)
  expect(page.events.find(e => e.kind === 'tool.host.responded')?.data).toMatchObject({ outcome: 'execution_failed' })
  expect(JSON.stringify(page)).not.toContain('PRIVATE-EXCEPTION')
  const invalid = a.start(c.contextId, 'fixture:invalid', { tools: [alpha], observation: opts }, () => { throw new Error('must not run') })
  await terminal(a, invalid)
  expect(a.observations(invalid).events.find(e => e.kind === 'tool.rejected')?.data).toMatchObject({ code: 'invalid_input' })
  const denied = a.start(c.contextId, 'fixture:foreign', { tools: [alpha], observation: opts }, () => { throw new Error('must not run') })
  await terminal(a, denied)
  expect(a.observations(denied).events.find(e => e.kind === 'tool.rejected')?.data).toMatchObject({ code: 'rejected' })
})


test('Pi native response usage is reported once by response ID; output and reasoning are not summed', () => {
  const { store, run, writer } = journal()
  const events = new CliEvents('pi', writer)
  const native = { type: 'message_end', message: { role: 'assistant', responseId: 'response-1', usage: { input: 12, output: 7, reasoning: 3, totalTokens: 19 }, content: [], stopReason: 'stop' } }
  events.push(JSON.stringify(native) + '\n' + JSON.stringify(native) + '\n')
  writer.finish('succeeded', true)
  const usage = readObservations(store, run).events.filter(e => e.kind === 'usage')
  expect(usage).toHaveLength(1)
  expect(usage[0]).toMatchObject({ native: { responseId: 'response-1' }, data: { scope: 'message', aggregation: 'delta', inputTokens: 12, outputTokens: 7, reasoningOutputTokens: 3, totalTokens: 19 } })
})

test('two concurrent executions keep independent trace and native-call namespaces', async () => {
  const { a, c } = cli('pi')
  const cwd = join(root, 'other-cwd'), configDir = join(root, 'other-config'), sessionDir = join(root, 'other-sessions')
  for (const d of [cwd, configDir, sessionDir]) mkdirSync(d)
  writeFileSync(join(configDir, 'auth.json'), '{"fixture":true}\n')
  const other = a.createContext(cwd, { configDir, sessionDir })
  const second = { ...opts, traceparent: '00-1af7651916cd43dd8448eb211c80319c-c7ad6b7169203331-01' }
  const ids = [
    a.start(c.contextId, 'fixture:reject', { tools: [alpha], observation: opts }, () => ({ ok: true, output: 'first' })),
    a.start(other.contextId, 'fixture:reject', { tools: [alpha], observation: second }, () => ({ ok: true, output: 'second' })),
  ]
  await Promise.all(ids.map(id => terminal(a, id)))
  for (const [i, id] of ids.entries()) {
    const page = a.observations(id)
    expect(page.status.complete).toBe(true)
    expect(page.events.every(e => e.runId === id && e.contextId === (i === 0 ? c.contextId : other.contextId) && e.traceId === (i === 0 ? opts : second).traceparent.slice(3, 35))).toBe(true)
  }
})

test('empty nested schemas preserve arbitrary transported JSON while roots remain typed', () => {
  const { assertHostTools, validateToolInput, writePiExtension } = require('../execution/hostTools') as typeof import('../execution/hostTools')
  const schema = { type: 'object' as const, properties: { slot: { type: 'string' as const }, value: {} }, required: ['slot', 'value'], additionalProperties: false }
  expect(() => assertHostTools([{ name: 'candidate', description: 'Write JSON', inputSchema: schema }])).not.toThrow()
  for (const value of [null, 1, 'text', [1, { x: true }], { any: ['JSON'] }]) expect(validateToolInput(schema, { slot: 'result', value })).toBe(true)
  expect(validateToolInput(schema, { slot: 'result' })).toBe(false)
  expect(() => assertHostTools([{ name: 'candidate', description: 'Bad root', inputSchema: {} }])).toThrow('invalid_request')
  expect(() => assertHostTools([{ name: 'candidate', description: 'Ambiguous', inputSchema: { type: 'object', properties: { value: { required: ['x'] } } } }])).toThrow('invalid_request')
  const extension = join(root, 'extension.mjs')
  writePiExtension(extension, [{ name: 'candidate', description: 'Write JSON', inputSchema: schema }], '/tmp/test-not-connected.sock', 'serial')
  expect(readFileSync(extension, 'utf8')).toContain('return Type.Unsafe(schema)')
})


test('string and numeric tool limits remain enforced, including Unicode scalar length', () => {
  const { assertHostTools, validateToolInput } = require('../execution/hostTools') as typeof import('../execution/hostTools')
  const schema = { type: 'object' as const, properties: { query: { type: 'string' as const, minLength: 1, maxLength: 2 }, count: { type: 'integer' as const, minimum: 1, maximum: 8 } }, required: ['query', 'count'], additionalProperties: false }
  expect(() => assertHostTools([{ name: 'search', description: 'Bounded search', inputSchema: schema }])).not.toThrow()
  expect(validateToolInput(schema, { query: '😀😀', count: 8 })).toBe(true)
  for (const input of [{ query: '', count: 1 }, { query: 'abc', count: 1 }, { query: 'ok', count: 9 }, { query: 'ok', count: 1.5 }]) expect(validateToolInput(schema, input)).toBe(false)
})

test('public purge is idempotent, scoped and preserves business and native state', async () => {
  const { a, c } = cli('pi')
  const id = a.start(c.contextId, 'fixture:tools', { tools: [alpha, beta], observation: opts }, () => ({ ok: true, output: 'safe' }))
  const result = await terminal(a, id)
  const original = a.observations(id)
  expect(original.events.length).toBeGreaterThan(0)
  const native = readFileSync(c.nativeSessionFile!, 'utf8'), auth = readFileSync(join(c.configDir!, 'auth.json'), 'utf8')
  expect(a.capabilities().observation?.purge).toBe(true)
  const clean = a.purgeObservations(id)
  expect(clean).toMatchObject({ state: 'purged', runId: id, contextId: c.contextId })
  expect(a.purgeObservations(id)).toEqual(clean)
  expect(a.observations(id, 5)).toMatchObject({ events: [], nextSequence: 5, status: { complete: false, fault: 'expired' } })
  expect(a.query(id)).toMatchObject({ status: result.status, stopped: result.stopped, output: result.output })
  expect(readFileSync(c.nativeSessionFile!, 'utf8')).toBe(native)
  expect(readFileSync(join(c.configDir!, 'auth.json'), 'utf8')).toBe(auth)
  for (const e of original.events.filter(e => e.kind === 'tool.requested')) expect(a.toolCall((e.data as { callId: string }).callId)).toBeDefined()
  expect(require('node:fs').existsSync(join(root, 'data/observations', `${id}.jsonl`))).toBe(false)
  expect(require('node:fs').existsSync(join(root, 'data/observations', `${id}.json`))).toBe(false)
})

test('purge refuses active, unconfirmed stop and pending host effects without deleting observations', () => {
  const { store, run, writer } = journal()
  const a = new ExecutionClient({ dataDir: store.root, connection: { contractVersion: 1, fields: { transport: 'agent-cli', runtime: 'pi' } } })
  store.write('runs', run.runId, run)
  expect(() => a.purgeObservations(run.runId)).toThrow('context_busy')
  run.finishedAt = Date.now(); run.status = 'unknown'; run.stopped = false; writer.finish('unknown', false); store.write('runs', run.runId, run)
  expect(() => a.purgeObservations(run.runId)).toThrow('context_busy')
  run.stopped = true; store.write('runs', run.runId, run)
  const callId = randomUUID()
  store.write('calls', callId, { version: 1, callId, contextId: run.contextId, runId: run.runId, status: 'pending' })
  expect(() => a.purgeObservations(run.runId)).toThrow('context_busy')
  expect(require('node:fs').existsSync(join(store.root, 'observations', `${run.runId}.jsonl`))).toBe(true)
  store.write('calls', callId, { version: 1, callId, contextId: run.contextId, runId: run.runId, status: 'reconciled' })
  expect(a.purgeObservations(run.runId).state).toBe('purged')
})

test('interrupted purge stays pending and retries even after partial deletion', () => {
  const { store, run, writer } = journal()
  writer.finish('succeeded', true); run.status = 'succeeded'; run.stopped = true; run.finishedAt = Date.now(); store.write('runs', run.runId, run)
  const a = new ExecutionClient({ dataDir: store.root, connection: { contractVersion: 1, fields: { transport: 'agent-cli', runtime: 'pi' } } })
  // A directory at the second expected file makes unlink fail after JSONL removal.
  const state = join(store.root, 'observations', `${run.runId}.json`)
  rmSync(state); mkdirSync(state)
  expect(() => a.purgeObservations(run.runId)).toThrow('storage_error')
  expect(a.observations(run.runId)).toMatchObject({ events: [], status: { fault: 'cleanup_pending', complete: false } })
  rmSync(state, { recursive: true })
  expect(a.purgeObservations(run.runId).state).toBe('purged')
  expect(a.observations(run.runId).status.fault).toBe('expired')
})

test('incremental source reads do not reparse historical records and still validate appended data', () => {
  const { store, run, writer } = journal({ ...opts, maxEvents: 3000, maxBytes: 4 * 1024 * 1024 })
  for (let n = 0; n < 2000; n++) writer.emit('agent.activity', { category: 'message' })
  readObservations(store, run, 2000, 1)
  const parse = jest.spyOn(JSON, 'parse')
  try {
    for (let n = 0; n < 5; n++) expect(readObservations(store, run, 2000, 1).events).toEqual([])
    // Only the small state JSON may be parsed on an empty incremental page.
    expect(parse.mock.calls.length).toBeLessThanOrEqual(5)
    writer.emit('agent.activity', { category: 'turn' })
    parse.mockClear()
    expect(readObservations(store, run, 2000, 1).events.map(e => e.sequence)).toEqual([2001])
    expect(parse.mock.calls.length).toBeLessThanOrEqual(3)
    writer.finish('succeeded', true)
    expect(readObservations(store, run, 2001, 1)).toMatchObject({ status: { complete: true }, nextSequence: 2002 })
    appendFileSync(join(store.root, 'observations', `${run.runId}.jsonl`), '{bad-tail')
    expect(readObservations(store, run, 2002, 1).status).toMatchObject({ complete: false, fault: 'corrupt' })
  } finally { parse.mockRestore() }
})

test('source indexes invalidate on replacement truncation and same-size edits', () => {
  const { renameSync, utimesSync } = require('node:fs') as typeof import('node:fs')
  const { store, run, writer } = journal()
  writer.emit('agent.activity', { category: 'message' }); writer.finish('succeeded', true)
  const file = join(store.root, 'observations', `${run.runId}.jsonl`), original = readFileSync(file, 'utf8')
  expect(readObservations(store, run).status.complete).toBe(true)
  expect(readObservations(store, { ...run, contextId: 'wrong-context' }).status).toMatchObject({ complete: false, fault: 'corrupt' })
  expect(readObservations(store, run).status.complete).toBe(true)
  const rows = original.trim().split('\n').map(line => JSON.parse(line))
  rows[0].id = rows[1].id
  const edited = rows.map(row => JSON.stringify(row)).join('\n') + '\n'
  expect(Buffer.byteLength(edited)).toBe(Buffer.byteLength(original))
  writeFileSync(file, edited); utimesSync(file, new Date(), new Date(Date.now() + 2000))
  expect(readObservations(store, run).status).toMatchObject({ complete: false, fault: 'corrupt' })
  writeFileSync(file + '.replacement', original); renameSync(file + '.replacement', file)
  expect(readObservations(store, run).status.complete).toBe(true)
  writeFileSync(file, original.slice(0, 20))
  expect(readObservations(store, run).status).toMatchObject({ complete: false, fault: 'corrupt' })
})

test('historical prefix edits followed by append are corrupt in both hot and cold reads', () => {
  const { store, run, writer } = journal()
  writer.emit('agent.activity', { category: 'message' })
  const file = join(store.root, 'observations', `${run.runId}.jsonl`)
  const original = readFileSync(file, 'utf8')
  expect(readObservations(store, run).events).toHaveLength(1)
  const changed = original.replace(run.contextId, randomUUID())
  expect(Buffer.byteLength(changed)).toBe(Buffer.byteLength(original))
  writeFileSync(file, changed)
  writer.finish('succeeded', true)
  const hot = readObservations(store, run)
  const cold = readObservations(new ExecutionStore(store.root), run)
  expect(hot.status).toMatchObject({ complete: false, fault: 'corrupt' })
  expect(cold.status).toMatchObject({ complete: false, fault: 'corrupt' })
  expect(hot.events.every(e => e.contextId === run.contextId)).toBe(true)
})
