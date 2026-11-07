#!/usr/bin/env node
// Deterministic app-server protocol fixture, never native Codex evidence.
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto'), readline = require('node:readline')
const args = process.argv.slice(2)
let scenario = {}
try { scenario = JSON.parse(fs.readFileSync(path.join(process.cwd(), 'codex-fixture.json'), 'utf8')) } catch {}
if (args.includes('--version')) { console.log(scenario.badVersion ? 'codex-cli 99.0.0' : 'codex-cli 0.166.0'); process.exit(0) }
if (args.includes('mcp') && args.includes('list')) {
  const locked = args.some(a => a.startsWith('mcp_servers='))
  console.log(JSON.stringify([{ name: 'inherited.with.dots', enabled: locked && !scenario.badInventory ? false : true, transport: { type: 'stdio' } }]))
  process.exit(0)
}
if (!args.includes('app-server')) process.exit(1)
fs.writeFileSync(path.join(process.cwd(), 'codex.pid'), String(process.pid))
fs.writeFileSync(path.join(process.cwd(), 'codex-launch.json'), JSON.stringify({ args, home: process.env.CODEX_HOME, credentialPresent: Object.keys(process.env).some(k => /_API_KEY$|_AUTH_TOKEN$/.test(k)) }))
let threadId, file, tools = [], memory = '', turnId, toolResponses = [], expected = 0
const emit = x => process.stdout.write(JSON.stringify(x) + '\n')
function complete(text) {
  const p = { threadId, turnId, item: { type: 'agentMessage', id: 'answer', text } }
  emit({ method: 'item/started', params: p }); emit({ method: 'item/completed', params: p })
  emit({ method: 'turn/completed', params: { threadId, turn: { id: turnId, status: 'completed' } } })
}
function nativeCall(name, n) {
  emit({ id: `native-request-${n}`, method: 'item/tool/call', params: { threadId: scenario.foreign ? crypto.randomUUID() : threadId, turnId, callId: `native-call-${n}`, tool: name, arguments: { n } } })
}
readline.createInterface({ input: process.stdin }).on('line', line => {
  const m = JSON.parse(line)
  if (!m.method) {
    if (String(m.id).startsWith('native-request-')) {
      toolResponses.push(m.result)
      if (toolResponses.length === expected) complete(JSON.stringify(toolResponses))
    }
    return
  }
  if (m.method === 'initialize') { emit({ id: m.id, result: scenario.badHandshake ? {} : { userAgent: 'codex-fixture/0.166.0' } }); return }
  if (m.method === 'initialized') return
  if (m.method === 'thread/start') {
    threadId = crypto.randomUUID(); tools = m.params.dynamicTools
    file = path.join(process.env.CODEX_HOME, 'sessions', `rollout-${threadId}.jsonl`)
    fs.writeFileSync(file, JSON.stringify({ type: 'session_meta', payload: { id: threadId, cwd: process.cwd() } }) + '\n')
    fs.appendFileSync(file, JSON.stringify({ type: 'fixture_state', tools, memory: '' }) + '\n')
  }
  if (m.method === 'thread/resume') {
    file = m.params.path
    if (!fs.existsSync(file)) { emit({ id: m.id, error: { code: -32600, message: 'session missing' } }); return }
    const lines = fs.readFileSync(file, 'utf8').trim().split('\n').map(JSON.parse)
    threadId = lines[0].payload.id
    const state = lines.filter(x => x.type === 'fixture_state').pop(); tools = state.tools; memory = state.memory
  }
  if (m.method === 'thread/start' || m.method === 'thread/resume') {
    emit({ id: m.id, result: { model: scenario.modelMismatch ? 'fallback-model' : m.params.model, cwd: process.cwd(), thread: { id: threadId, path: file, cwd: process.cwd(), ephemeral: false } } }); return
  }
  if (m.method === 'turn/start') {
    if (scenario.authError) { emit({ id: m.id, error: { code: -32000, message: '401 Authentication failed DO-NOT-LEAK-SECRET' } }); return }
    turnId = crypto.randomUUID()
    emit({ id: m.id, result: { turn: { id: turnId } } })
    emit({ method: 'turn/started', params: { threadId, turn: { id: turnId, status: 'inProgress' } } })
    const input = m.params.input[0].text
    if (scenario.nativeTool) { emit({ method: 'item/started', params: { threadId, turnId, item: { id: 'native-bash', type: 'commandExecution' } } }); return }
    if (input === 'fixture:sleep') return
    if (input === 'fixture:tools' || input === 'fixture:hold') {
      expected = input === 'fixture:tools' ? 2 : 1
      nativeCall(tools[0].name, 1)
      if (scenario.earlyComplete) complete('premature')
      if (expected === 2) nativeCall(tools[1].name, 2)
      return
    }
    if (input === 'fixture:revoked') { expected = 1; nativeCall(tools[0].name, 1); return }
    const answer = input === 'fixture:recall' ? memory : input
    if (input !== 'fixture:recall') { memory = input; fs.appendFileSync(file, JSON.stringify({ type: 'fixture_state', tools, memory }) + '\n') }
    complete(answer)
  }
})
// Keep the simulated app-server alive just as the real server waits for requests.
