import { mkdtempSync, mkdirSync, symlinkSync, chmodSync, writeFileSync, readFileSync, existsSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { createToolDispatcher } from '../execution/hostTools'
import { ExecutionStore } from '../execution/store'
import { randomUUID } from 'node:crypto'
const { ExecutionClient } = require('../../dist/execution/ExecutionClient') as typeof import('../execution/ExecutionClient')
const { disabledCodexMcp } = require('../../dist/execution/codex') as typeof import('../execution/codex')
let root: string, seq: number, env: NodeJS.ProcessEnv
const connection = { contractVersion: 1, fields: { transport: 'agent-cli', runtime: 'codex', model: 'test-model' } }
const alpha = { name: 'alpha', description: 'Alpha', inputSchema: { type: 'object' as const, properties: { n: { type: 'integer' as const } }, required: ['n'], additionalProperties: false } }
const beta = { ...alpha, name: 'beta' }
beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'milkie-codex-')); seq = 0
  mkdirSync(join(root, 'bin'))
  const fixture = resolve('tests/fixtures/codex-cli.cjs'); chmodSync(fixture, 0o755)
  symlinkSync(fixture, join(root, 'bin', 'codex'))
  env = { ...process.env, PATH: `${join(root, 'bin')}:${process.env.PATH}`, OPENAI_API_KEY: 'DO-NOT-INHERIT' }
})
afterEach(() => rmSync(root, { recursive: true, force: true }))
function client() { return new ExecutionClient({ dataDir: join(root, 'data'), connection, env, lifetime: 'host' }) }
function context(a = client(), scenario: Record<string, unknown> = {}) {
  const cwd = join(root, `cwd-${++seq}`), configDir = join(root, `config-${seq}`), sessionDir = join(root, `sessions-${seq}`)
  for (const d of [cwd, configDir, sessionDir]) mkdirSync(d)
  writeFileSync(join(configDir, 'auth.json'), '{"fixture":true}\n')
  writeFileSync(join(cwd, 'codex-fixture.json'), JSON.stringify(scenario))
  return a.createContext(cwd, { configDir, sessionDir })
}
async function terminal(a: ReturnType<typeof client>, runId: string) {
  const end = Date.now() + 15000
  while (Date.now() < end) { const r = a.query(runId); if (r?.finishedAt) return r; await delay(30) }
  throw new Error('No Codex terminal record.')
}

test('Codex execution and exact resume preserve native session and dedicated storage', async () => {
  const a = client(), c = context(a)
  expect(a.capabilities()).toMatchObject({ supported: true, resume: true, nativeCallId: true, modelIterations: false, toolPolicies: ['read-only'], availability: 'unchecked' })
  const first = await terminal(a, a.start(c.contextId, 'unique-codex-marker'))
  expect(first).toMatchObject({ status: 'succeeded', output: 'unique-codex-marker', stopped: true })
  const b = client(), resumed = await terminal(b, b.start(c.contextId, 'fixture:recall'))
  expect(resumed).toMatchObject({ status: 'succeeded', output: 'unique-codex-marker', nativeSessionId: first.nativeSessionId })
  const launch = JSON.parse(readFileSync(join(c.cwd, 'codex-launch.json'), 'utf8'))
  expect(launch.home).toBe(c.configDir); expect(launch.credentialPresent).toBe(false)
  expect(launch.args).toContain('features.shell_tool=false')
  expect(launch.args).toContain('features.multi_agent=false')
  expect(launch.args).toContain('mcp_servers={"inherited.with.dots"={command="/usr/bin/false",enabled=false}}')
  expect(launch.args.join(' ')).not.toContain('unique-codex-marker')
  rmSync(a.getContext(c.contextId).nativeSessionFile!)
  expect(() => a.start(c.contextId, 'must-not-start-fresh')).toThrow('session_missing')
})

test.each(['serial', 'parallel'] as const)('%s dynamic tool forwarding retains native ids and results', async forwarding => {
  const a = client(), c = context(a), seen: string[] = []
  let active = 0, max = 0
  const runId = a.start(c.contextId, 'fixture:tools', { tools: [alpha, beta], forwarding }, async call => {
    active++; max = Math.max(max, active); seen.push(call.nativeCallId!)
    await delay(80); active--
    return call.name === 'alpha' ? { ok: true, output: 'alpha-result' } : { ok: false, code: 'rejected', message: 'beta-denied' }
  })
  const result = await terminal(a, runId)
  expect(result).toMatchObject({ status: 'succeeded', stopped: true })
  expect(seen).toEqual(['native-call-1', 'native-call-2'])
  expect(max).toBe(forwarding === 'parallel' ? 2 : 1)
  expect(JSON.parse(result.output!)).toEqual(expect.arrayContaining([
    { contentItems: [{ type: 'inputText', text: 'alpha-result' }], success: true },
    { contentItems: [{ type: 'inputText', text: 'rejected: beta-denied' }], success: false },
  ]))
})

test.each(['global', 'omitted', 'targeted'])('a %s warning after tool delivery does not change execution ownership', async warning => {
  const a = client(), c = context(a, { warning })
  let effects = 0
  const r = await terminal(a, a.start(c.contextId, 'fixture:hold', { tools: [alpha] }, () => {
    effects++; return { ok: true, output: 'accepted-once' }
  }))
  expect(r).toMatchObject({ status: 'succeeded', stopped: true })
  expect(effects).toBe(1)
  expect(r.output).toContain('accepted-once')
  expect(JSON.stringify(r)).not.toContain('DO-NOT-LEAK-WARNING')
})

test.each([{ warning: 'foreign' }, { warning: 'global', warningRequest: true }, { warning: 'global', warningTurn: true }])('invalid warning scope %j still fails after the authorized tool result', async scenario => {
  const a = client(), c = context(a, scenario)
  let effects = 0
  const r = await terminal(a, a.start(c.contextId, 'fixture:hold', { tools: [alpha] }, () => {
    effects++; return { ok: true, output: 'accepted-once' }
  }))
  expect(r).toMatchObject({ status: 'failed', stopped: true, code: 'session_mismatch' })
  expect(effects).toBe(1)
  expect(JSON.stringify(r)).not.toContain('DO-NOT-LEAK-WARNING')
})

test('resume rejects new/changed tools before launch and revoked tools never reach the host', async () => {
  const a = client(), c = context(a)
  expect((await terminal(a, a.start(c.contextId, 'first', { tools: [alpha, beta] }, () => ({ ok: true, output: 'ok' })))).status).toBe('succeeded')
  expect(() => a.start(c.contextId, 'x', { tools: [{ ...alpha, description: 'changed' }] }, () => ({ ok: true, output: 'x' }))).toThrow('unsupported_constraint')
  expect(() => a.start(c.contextId, 'x', { tools: [{ ...alpha, name: 'new_tool' }] }, () => ({ ok: true, output: 'x' }))).toThrow('unsupported_constraint')
  let effects = 0
  const r = await terminal(a, a.start(c.contextId, 'fixture:revoked', { tools: [beta] }, () => { effects++; return { ok: true, output: 'must-not-run' } }))
  expect(r.status).toBe('succeeded'); expect(effects).toBe(0)
  expect(JSON.parse(r.output!)[0]).toMatchObject({ success: false, contentItems: [{ text: 'rejected: Tool is not registered.' }] })
})

test.each([
  [{ badVersion: true }, 'unsupported_version'],
  [{ badHandshake: true }, 'protocol_error'],
  [{ badInventory: true }, 'policy_mismatch'],
  [{ modelMismatch: true }, 'policy_mismatch'],
  [{ nativeTool: true }, 'policy_mismatch'],
  [{ authError: true }, 'auth_failed'],
  [{ foreign: true }, 'session_mismatch'],
  [{ nullToolThread: true }, 'session_mismatch'],
  [{ wrongToolTurn: true }, 'session_mismatch'],
] as const)('native failure %j is explicit and has no host effects', async (scenario, code) => {
  const a = client(), c = context(a, scenario)
  let effects = 0
  const r = await terminal(a, a.start(c.contextId, 'fixture:hold', { tools: [alpha] }, () => { effects++; return { ok: true, output: 'x' } }))
  expect(r).toMatchObject({ status: 'failed', stopped: true, code })
  expect(effects).toBe(0); expect(JSON.stringify(r)).not.toContain('DO-NOT-LEAK')
})

test('unsupported constraints reject before any native launch; cancellation and timeout confirm stop', async () => {
  const a = client(), c = context(a)
  expect(() => a.start(c.contextId, 'x', { maxModelIterations: 2 })).toThrow('unsupported_constraint')
  expect(() => a.start(c.contextId, 'x', { toolPolicy: 'standard' })).toThrow('unsupported_constraint')
  expect(existsSync(join(c.cwd, 'codex.pid'))).toBe(false)
  const id = a.start(c.contextId, 'fixture:sleep')
  for (let n = 0; n < 150 && !existsSync(join(c.cwd, 'codex.pid')); n++) await delay(20)
  expect(existsSync(join(c.cwd, 'codex.pid'))).toBe(true)
  const pid = Number(readFileSync(join(c.cwd, 'codex.pid'), 'utf8'))
  expect(await a.cancel(id)).toMatchObject({ status: 'cancelled', stopped: true })
  expect(() => process.kill(pid, 0)).toThrow()
  const t = context(a)
  expect(await terminal(a, a.start(t.contextId, 'fixture:sleep', { timeoutMs: 500 }))).toMatchObject({ status: 'timed_out', stopped: true })
})

test('a host result not delivered to the native transport stays pending', async () => {
  const a = client(), c = context(a), store = new ExecutionStore(join(root, 'data')), runId = randomUUID()
  const dispatch = createToolDispatcher({ tools: [alpha], forwarding: 'serial', contextId: c.contextId, runId, store, alive: () => true, onHost: () => Promise.resolve({ ok: true, output: 'effect-done' }) })
  await dispatch({ id: 'native-request', name: 'alpha', nativeCallId: 'native-id', input: { n: 1 } }, 64, () => Promise.resolve(false))
  expect(a.pendingToolCalls(c.contextId)).toEqual([expect.objectContaining({ runId, status: 'pending', nativeCallId: 'native-id' })])
  expect(() => a.start(c.contextId, 'must-not-replay')).toThrow('context_busy')
})

test('MCP inventory includes literal names and unknown transports fail closed', () => {
  expect(disabledCodexMcp([{ name: 'a."b', transport: { type: 'streamable_http' } }])).toBe('mcp_servers={"a.\\"b"={url="http://127.0.0.1:9",enabled=false}}')
  expect(() => disabledCodexMcp([{ name: 'x', transport: { type: 'unknown' } }])).toThrow('policy_mismatch')
})

test('missing native binary is unavailable and never silently selects another runtime', async () => {
  const a = new ExecutionClient({ dataDir: join(root, 'data'), connection, env: { ...env, PATH: join(root, 'empty-bin') }, lifetime: 'host' })
  const c = context(a)
  expect(await terminal(a, a.start(c.contextId, 'must-not-fallback'))).toMatchObject({ status: 'failed', stopped: true, code: 'runtime_missing' })
  expect(existsSync(join(c.cwd, 'codex.pid'))).toBe(false)
})


test('native completion cannot mark an unresolved host tool as succeeded', async () => {
  const a = client(), c = context(a, { earlyComplete: true })
  const r = await terminal(a, a.start(c.contextId, 'fixture:hold', { tools: [alpha], timeoutMs: 1500 }, () => new Promise(() => undefined)))
  expect(r).toMatchObject({ status: 'timed_out', stopped: true })
  expect(r.output).toBeUndefined()
  expect(a.pendingToolCalls(c.contextId)).toEqual([expect.objectContaining({ runId: r.runId, status: 'pending' })])
})


test.each(['cached', 'live'] as const)('explicit %s search exports only native-owned results and resumes unchanged', async webSearch => {
  const a=client(), c=context(a)
  const id=a.start(c.contextId,'fixture:search',{webSearch,observation:{traceparent:'00-'+'1'.repeat(32)+'-'+'2'.repeat(16)+'-01',maxEvents:100,maxBytes:65536}})
  expect(await terminal(a,id)).toMatchObject({status:'succeeded',stopped:true,webSearch})
  expect(client().webSearches(id)).toMatchObject({runId:id,contextId:c.contextId,mode:webSearch,complete:true,items:[{nativeItemId:'native-search-1',phase:'completed',query:'sensitive-search-query',results:[{url:'https://example.com/source',content:'sensitive-search-result'}]}]})
  const trace=JSON.stringify(a.observations(id))
  expect(trace).not.toContain('sensitive-search')
  expect(trace).toContain('agent.activity')
  const launch=JSON.parse(readFileSync(join(c.cwd,'codex-launch.json'),'utf8'))
  expect(launch.args).toContain('web_search='+JSON.stringify(webSearch))
  expect(launch.args).toContain('features.shell_tool=false')
  expect(launch.args).toContain('features.multi_agent=false')
  expect(() => client().start(c.contextId,'x')).toThrow('unsupported_constraint')
  const resumed=a.start(c.contextId,'fixture:search',{webSearch})
  expect(await terminal(a,resumed)).toMatchObject({status:'succeeded',stopped:true})
  expect(a.webSearches(resumed)?.runId).toBe(resumed)
  expect(a.webSearches(id)?.runId).toBe(id)
})

test('default search remains disabled and cannot be enabled by a native event', async () => {
  const a=client(),c=context(a),id=a.start(c.contextId,'fixture:search')
  expect(await terminal(a,id)).toMatchObject({status:'failed',code:'policy_mismatch',stopped:true})
  expect(a.webSearches(id)).toMatchObject({mode:'disabled',items:[]})
  expect(() => a.start(c.contextId,'x',{webSearch:'live'})).toThrow('unsupported_constraint')
  expect(() => a.start(c.contextId,'x',{webSearch:'invalid' as any})).toThrow('unsupported_constraint')
})

test.each([{foreignSearch:true},{malformedSearch:true},{largeSearch:true}])('search protocol/identity/capacity failure %j is explicit', async scenario => {
  const a=client(),c=context(a,scenario),id=a.start(c.contextId,'fixture:search',{webSearch:'live'})
  expect(await terminal(a,id)).toMatchObject({status:'failed',stopped:true,code:scenario.foreignSearch?'session_mismatch':scenario.malformedSearch?'protocol_error':'native_search_limit_exceeded'})
  expect(a.webSearches(id)?.items.every(x=>x.results===null)).toBe(true)
  expect(a.webSearches(id)?.complete).toBe(false)
})

test('null native results stay unknown and cancelled search stays started', async () => {
  const a=client(),c=context(a,{nullSearch:true}),id=a.start(c.contextId,'fixture:search',{webSearch:'live'})
  expect(await terminal(a,id)).toMatchObject({status:'succeeded',stopped:true})
  expect(a.webSearches(id)).toMatchObject({complete:true,items:[{results:null}]})
  const d=context(a),sleep=a.start(d.contextId,'fixture:search-sleep',{webSearch:'live'})
  for(let i=0;i<200&&a.webSearches(sleep)?.items.length===0;i++)await delay(20)
  expect(a.webSearches(sleep)?.items).toHaveLength(1)
  expect(await a.cancel(sleep)).toMatchObject({status:'cancelled',stopped:true})
  expect(client().webSearches(sleep)).toMatchObject({complete:false,items:[{phase:'started',results:null}]})
  rmSync(join(root,'data','native-searches',id+'.json'))
  expect(()=>a.webSearches(id)).toThrow('storage_error')
})


test('native shell is still rejected with explicit web search authorization', async () => {
 const a=client(),c=context(a,{nativeTool:true})
 expect(await terminal(a,a.start(c.contextId,'x',{webSearch:'live'}))).toMatchObject({status:'failed',code:'policy_mismatch',stopped:true})
})


test('the 65th native search fails without silently truncating a complete result', async () => {
 const a=client(),c=context(a,{searchCount:65}),id=a.start(c.contextId,'fixture:search',{webSearch:'live'})
 expect(await terminal(a,id)).toMatchObject({status:'failed',code:'native_search_limit_exceeded',stopped:true})
 expect(a.webSearches(id)).toMatchObject({complete:false})
 expect(a.webSearches(id)?.items).toHaveLength(64)
})
