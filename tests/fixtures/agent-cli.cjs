#!/usr/bin/env node
// Deterministic protocol fixture, not evidence for native CLI support.
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto')
const args = process.argv.slice(2), pi = args.includes('--print')
const get = key => args[args.indexOf(key) + 1]
function configuredMcpLaunch() {
  const override = path.join(process.cwd(), '.grok', 'fixture-mcp-args.json')
  if (fs.existsSync(override)) {
    const body = JSON.parse(fs.readFileSync(override, 'utf8'))
    return { command: typeof body.command === 'string' ? body.command : configuredMcpCommand(), args: Array.isArray(body.args) ? body.args : [] }
  }
  const files = [path.join(process.cwd(), '.grok', 'config.toml'), path.join(process.env.GROK_HOME || '', 'config.toml')]
  for (const file of files) {
    if (!fs.existsSync(file)) continue
    const text = fs.readFileSync(file, 'utf8')
    const command = text.match(/\[mcp_servers\.milkie\][\s\S]*?\ncommand\s*=\s*"([^"]+)"/)
    const args = text.match(/\[mcp_servers\.milkie\][\s\S]*?\nargs\s*=\s*\[([\s\S]*?)\]/)
    if (!command || !args) continue
    return { command: command[1], args: [...args[1].matchAll(/"([^"]*)"/g)].map(match => match[1]) }
  }
  return { command: process.execPath, args: [] }
}
if (args[0] === 'mcp' && args.includes('list')) {
  const launch = configuredMcpLaunch()
  process.stdout.write(JSON.stringify([{ name: 'milkie', command: launch.command, args: launch.args, enabled: true }]))
  process.exit(0)
}
function configuredMcpCommand() {
  const files = [path.join(process.cwd(), '.grok', 'config.toml'), path.join(process.env.GROK_HOME || '', 'config.toml')]
  for (const file of files) {
    if (!fs.existsSync(file)) continue
    const match = fs.readFileSync(file, 'utf8').match(/\[mcp_servers\.milkie\][\s\S]*?\ncommand\s*=\s*"([^"]+)"/)
    if (match) return match[1]
  }
  return process.execPath
}
if (args.includes('inspect')) {
  const vendors = ['CURSOR', 'CLAUDE', 'CODEX'], surfaces = ['SKILLS', 'RULES', 'AGENTS', 'MCPS', 'HOOKS', 'SESSIONS']
  const cells = []
  for (const vendor of vendors) for (const surface of surfaces) cells.push({ vendor, surface, enabled: process.env[`GROK_${vendor}_${surface}_ENABLED`] === '0' ? false : true, source: 'env' })
  const agents = [{ name: 'general-purpose', source: { type: 'builtin' } }, { name: 'explore', source: { type: 'builtin' } }, { name: 'plan', source: { type: 'builtin' } }]
  if (fs.existsSync(path.join(process.cwd(), '.grok', 'agents'))) agents.push({ name: 'evil', source: { type: 'project' } })
  process.stdout.write(JSON.stringify({
    mcpServers: [{ name: 'milkie', target: configuredMcpCommand() }, ...(fs.existsSync(path.join(process.cwd(), '.mcp.json')) ? [{ name: 'evil', target: 'evil' }] : [])],
    hooks: [], skills: [], plugins: [], lspServers: [], marketplaces: [], agents,
    externalCompat: { cells },
    permissions: { managedSettingsActive: process.env.GROK_MANAGED_CONFIG === '0' && process.env.GROK_MANAGED_MCPS_ENABLED === '0' ? false : true },
  }))
  process.exit(0)
}
let input = pi ? '' : fs.readFileSync(get('--prompt-file'), 'utf8')
function emit(e) { process.stdout.write(JSON.stringify(e) + '\n') }
async function run() {
  fs.writeFileSync(path.join(process.cwd(), 'runner.pid'), String(process.pid))
  const credential = Object.keys(process.env).find(key => /(?:^GROK_AUTH$|_API_KEY$|_AUTH_TOKEN$|_OAUTH_TOKEN$|_ACCESS_TOKEN$|_REFRESH_TOKEN$)/.test(key) && process.env[key])
  fs.writeFileSync(path.join(process.cwd(), 'cli-invocation.json'), JSON.stringify({
    grokHome: process.env.GROK_HOME ?? null,
    leaderSocket: args.includes('--leader-socket') ? get('--leader-socket') : null,
    piConfig: process.env.PI_CODING_AGENT_DIR ?? null,
    sessionDir: args.includes('--session-dir') ? get('--session-dir') : null,
    session: args.includes('--session') ? get('--session') : null,
    credentialPresent: Boolean(credential),
    home: process.env.HOME ?? null,
    envKeys: Object.keys(process.env).sort(),
    args,
  }))
  if (!pi && !process.env.GROK_HOME) { process.stderr.write('GROK_HOME missing'); process.exit(1) }
  const file = pi ? get('--session') : path.join(process.env.GROK_HOME, 'sessions', encodeURIComponent(fs.realpathSync(process.cwd())), get(args.includes('--resume') ? '--resume' : '--session-id'), 'chat_history.jsonl')
  let history
  if (fs.existsSync(file)) history = fs.readFileSync(file, 'utf8').trim().split('\n').map(JSON.parse)
  else history = [{ type: 'session', id: pi ? crypto.randomUUID() : get('--session-id'), cwd: fs.realpathSync(process.cwd()) }]
  if (input === 'fixture:auth') { process.stderr.write('Authentication failed token=DO-NOT-LEAK'); process.exit(1) }
  if (input === 'fixture:malformed') { process.stdout.write('not json\n'); process.exit(0) }
  fs.mkdirSync(path.dirname(file), {recursive:true})
  history.push({ input })
  fs.writeFileSync(file, history.map(JSON.stringify).join('\n') + '\n')
  const toolInputs = new Set(['fixture:tools', 'fixture:invalid', 'fixture:reject', 'fixture:foreign', 'fixture:hold', 'fixture:revoked'])
  if (toolInputs.has(input)) {
    if (!process.env.MILKIE_TOOL_SOCKET) { process.stderr.write('tool socket missing'); process.exit(1) }
    const call = (id, name, value) => ({ id, name, input: value, ...(pi ? { nativeCallId: `native-${name}` } : {}) })
    const calls = input === 'fixture:invalid' ? [call('1', 'alpha', { n: 'nope' })]
      : input === 'fixture:foreign' ? [call('1', 'bash', {})]
      : input === 'fixture:tools' ? [call('1', 'alpha', { n: 1 }), call('2', 'beta', { n: 2 })]
      : input === 'fixture:revoked' ? [call('1', 'alpha', { n: 1 })]
      : [call('1', 'alpha', { n: 1 })]
    const results = await new Promise((resolve, reject) => {
      const socket = require('node:net').connect(process.env.MILKIE_TOOL_SOCKET)
      let buf = '', got = []
      socket.on('error', reject)
      socket.on('data', chunk => {
        buf += chunk
        let nl
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl); buf = buf.slice(nl + 1)
          if (line) got.push(JSON.parse(line))
          if (input !== 'fixture:hold' && got.length === calls.length) { socket.end(); resolve(got) }
        }
      })
      socket.on('connect', () => {
        if (input === 'fixture:hold' && pi) emit(history[0])
        socket.write(calls.map(item => JSON.stringify(item)).join('\n') + '\n')
        if (input === 'fixture:hold') setInterval(() => {}, 1000)
      })
    })
    if (input === 'fixture:hold') return
    const output = results.map(item => item.ok ? item.output : item.code).join('|')
    const id = history[0].id
    if (pi) {
      emit(history[0])
      emit({ type: 'message_end', message: { role: 'assistant', content: [{ type: 'text', text: output }], stopReason: 'stop' } })
      emit({ type: 'agent_end', messages: [] })
    } else { emit({ type: 'text', data: output }); emit({ type: 'end', stopReason: 'end_turn', sessionId: id }) }
    return
  }
  if (pi) emit(history[0])
  if (input === 'fixture:sleep' || input === 'fixture:child' || input === 'fixture:escaped-child') {
    if (input === 'fixture:child' || input === 'fixture:escaped-child') {
      const child = require('node:child_process').spawn(process.execPath, ['-e', 'setInterval(()=>{},1000)'], {stdio:'ignore', detached: input === 'fixture:escaped-child'})
      fs.writeFileSync(path.join(process.cwd(), 'child.pid'), String(child.pid))
      // Reap the task child on TERM; the process-group stop must include it.
      process.on('SIGTERM', () => child.once('exit', () => process.exit(0)))
    }
    setInterval(()=>{},1000); return
  }
  const output = input === 'fixture:recall' ? history.filter(e=>e.input && !e.input.startsWith('fixture:')).map(e=>e.input).join('|') : input === 'fixture:cwd' ? process.cwd() : input
  const id = input === 'fixture:wrong-session' ? crypto.randomUUID() : history[0].id
  if (pi) {
    if (input === 'fixture:wrong-session') emit({type:'session',id})
    emit({type:'message_end',message:{role:'assistant',content:[{type:'text',text:output}],stopReason:input === 'fixture:json-error' ? 'error' : 'stop', errorMessage:'Authentication failed DO-NOT-LEAK'}})
    emit({type:'agent_end',messages:[]})
  } else { emit({type:'text',data:output}); emit({type:'end',stopReason:'end_turn',sessionId:id}) }
}
if (pi) { process.stdin.setEncoding('utf8');process.stdin.on('data',c=>input+=c);process.stdin.on('end',run) } else run()
