#!/usr/bin/env node
// Deterministic protocol fixture, not evidence for native CLI support.
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto')
const args = process.argv.slice(2), pi = args.includes('--print')
const get = key => args[args.indexOf(key) + 1]
let input = pi ? '' : fs.readFileSync(get('--prompt-file'), 'utf8')
function emit(e) { process.stdout.write(JSON.stringify(e) + '\n') }
function run() {
  fs.writeFileSync(path.join(process.cwd(), 'runner.pid'), String(process.pid))
  const credential = Object.keys(process.env).find(key => /(?:^GROK_AUTH$|_API_KEY$|_AUTH_TOKEN$|_OAUTH_TOKEN$|_ACCESS_TOKEN$|_REFRESH_TOKEN$)/.test(key) && process.env[key])
  fs.writeFileSync(path.join(process.cwd(), 'cli-invocation.json'), JSON.stringify({
    grokHome: process.env.GROK_HOME ?? null,
    leaderSocket: args.includes('--leader-socket') ? get('--leader-socket') : null,
    piConfig: process.env.PI_CODING_AGENT_DIR ?? null,
    sessionDir: args.includes('--session-dir') ? get('--session-dir') : null,
    session: args.includes('--session') ? get('--session') : null,
    credentialPresent: Boolean(credential),
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
