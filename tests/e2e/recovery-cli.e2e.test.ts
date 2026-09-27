/** Real CLI entry and real serve process against a loopback-only model endpoint. */
import http from 'http'
import type { AddressInfo } from 'net'
import { spawn, type ChildProcess } from 'child_process'
import fs from 'fs'
import path from 'path'
import os from 'os'

const entry = path.resolve(__dirname, '../../dist/cli/index.js')
function stop(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve()
  return new Promise(resolve => {
    const timer = setTimeout(() => child.kill('SIGKILL'), 3000)
    child.once('exit', () => { clearTimeout(timer); resolve() })
    child.kill('SIGTERM')
  })
}

test('CLI run/interrupt/resume and real serve preserve independent recovery records and terminal fields', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-recovery-cli-'))
  let mode: 'loop' | 'text' = 'loop'
  let requests = 0
  const model = http.createServer((req, res) => {
    let body = ''
    req.on('data', chunk => { body += chunk })
    req.on('end', () => {
      requests++
      const stream = JSON.parse(body).stream
      const call = { id: 'think-1', type: 'function', function: { name: 'think', arguments: '{"thoughts":"fixture"}' } }
      if (stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream' })
        const delta = mode === 'loop' ? { tool_calls: [{ index: 0, ...call }] } : { content: 'done' }
        res.write(`data: ${JSON.stringify({ id: 'fixture', choices: [{ index: 0, delta, finish_reason: null }] })}\n\n`)
        res.write(`data: ${JSON.stringify({ id: 'fixture', choices: [{ index: 0, delta: {}, finish_reason: mode === 'loop' ? 'tool_calls' : 'stop' }] })}\n\n`)
        res.end('data: [DONE]\n\n')
      } else {
        res.writeHead(200, { 'content-type': 'application/json' })
        res.end(JSON.stringify({ id: 'fixture', choices: [{ index: 0, message: mode === 'loop'
          ? { role: 'assistant', content: null, tool_calls: [call] } : { role: 'assistant', content: 'done' },
          finish_reason: mode === 'loop' ? 'tool_calls' : 'stop' }] }))
      }
    })
  })
  await new Promise<void>(resolve => model.listen(0, '127.0.0.1', resolve))
  const baseUrl = `http://127.0.0.1:${(model.address() as AddressInfo).port}/v1`
  const env = { ...process.env, OPENAI_API_KEY: 'fixture', VOLCENGINE_TOKEN: 'fixture', VOLCENGINE_API_BASE: baseUrl, LOG_LEVEL: 'silent' }
  const dataDir = path.join(dir, '.milkie')
  fs.mkdirSync(dataDir)
  const agentFile = path.join(dir, 'worker.md')
  fs.writeFileSync(agentFile, `---\nagentId: worker\nfsm:\n  states:\n    - name: react\n      type: llm\n      max_iterations: 1\nmodel:\n  provider: fixture\n  model: fixture\n  adapter: openai-compatible\n  baseUrl: ${baseUrl}\n---\nfixture`)
  fs.writeFileSync(path.join(dataDir, 'agents.json'), JSON.stringify({ agents: [{ id: 'worker', file: '../worker.md' }] }))
  const children: ChildProcess[] = []
  async function cli(args: string[]) {
    const proc = spawn(process.execPath, [entry, ...args], { cwd: dir, env, stdio: ['pipe', 'pipe', 'pipe'] })
    children.push(proc)
    return new Promise<{ code: number | null; stdout: string; stderr: string }>((resolve, reject) => {
      let stdout = '', stderr = ''
      const timer = setTimeout(() => { proc.kill('SIGKILL'); reject(new Error('CLI timeout')) }, 15000)
      proc.stdout.on('data', b => { stdout += b.toString() })
      proc.stderr.on('data', b => { stderr += b.toString() })
      proc.once('error', e => { clearTimeout(timer); reject(e) })
      proc.once('exit', code => { clearTimeout(timer); resolve({ code, stdout, stderr }) })
      proc.stdin.end()
    })
  }
  const events = (id: string) => fs.readFileSync(path.join(dataDir, 'runs', `${id}.jsonl`), 'utf8').trim().split('\n').map(s => JSON.parse(s))
  function assertTerminal(frame: Record<string, unknown>) {
    const terminal = events(String(frame.runId)).find(e => e.type === 'agent.run.completed').payload
    for (const key of ['status', 'stopReason', 'stopCode', 'partial', 'checkpointId', 'artifacts']) expect(frame[key]).toEqual(terminal[key])
  }
  try {
    const first = await cli(['agent', 'run', 'worker', '--input', 'i', '--context-id', 'cli-case'])
    expect(first.code).toBe(0)
    const original = JSON.parse(first.stdout)
    expect(original.stopReason).toBe('budget_exhausted')
    assertTerminal(original)
    const interrupted = await cli(['agent', 'interrupt', 'cli-case'])
    expect(interrupted.code).toBe(0)
    expect(JSON.parse(interrupted.stdout).status).toBe('interrupt-signaled')
    const paused = JSON.parse((await cli(['agent', 'resume', 'cli-case'])).stdout)
    expect(paused.runId).not.toBe(original.runId)
    expect(paused.stopReason).toBe('interrupted')
    assertTerminal(paused)
    mode = 'text'
    const resumed = JSON.parse((await cli(['agent', 'resume', 'cli-case'])).stdout)
    expect(resumed.stopReason).toBe('model_stop')
    expect(events(resumed.runId).find(e => e.type === 'agent.run.started').payload).toMatchObject({ previousRunId: paused.runId, resumedFromCheckpointId: paused.checkpointId })
    assertTerminal(resumed)

    const proc = spawn(process.execPath, [entry, 'serve', '--agent', agentFile, '--port', '0', '--state-store', 'sqlite', '--data-dir', dataDir], { cwd: dir, env, stdio: ['pipe', 'pipe', 'pipe'] })
    children.push(proc)
    const port = await new Promise<number>((resolve, reject) => {
      let output = ''
      const timer = setTimeout(() => reject(new Error(`serve timeout: ${output}`)), 15000)
      proc.stdout.on('data', b => { output += b.toString(); const m = output.match(/MILKIE_SERVE_READY (\d+)/); if (m) { clearTimeout(timer); resolve(Number(m[1])) } })
      proc.stderr.on('data', b => { output += b.toString() })
      proc.once('error', e => { clearTimeout(timer); reject(e) })
      proc.once('exit', code => { clearTimeout(timer); reject(new Error(`serve exited ${code}: ${output}`)) })
    })
    const base = `http://127.0.0.1:${port}`
    expect(await (await fetch(base + '/health')).json()).toEqual({ ok: true })
    const post = (route: string) => fetch(base + route, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ contextId: 'http-case', input: 'i' }), signal: AbortSignal.timeout(10000) })
    async function turn(route: string) {
      const response = await post(route)
      expect(response.status).toBe(200)
      const frames = (await response.text()).split('\n\n').filter(s => s.startsWith('event: agent.run.completed\n'))
      expect(frames).toHaveLength(1)
      const frame = JSON.parse(frames[0]!.split('\ndata: ')[1]!)
      assertTerminal(frame)
      return frame
    }
    mode = 'loop'
    const httpFirst = await turn('/chat')
    expect(httpFirst.stopReason).toBe('budget_exhausted')
    expect(await (await post('/interrupt')).json()).toMatchObject({ signaled: true })
    expect(await (await post('/context/state')).json()).toMatchObject({ exists: true, interruptSignaled: true })
    expect((await turn('/resume')).stopReason).toBe('interrupted')
    mode = 'text'
    const httpResumed = await turn('/resume')
    expect(httpResumed.stopReason).toBe('model_stop')
    expect(httpResumed.runId).not.toBe(httpFirst.runId)
    expect(requests).toBeGreaterThanOrEqual(4)
  } finally {
    await Promise.all(children.map(stop))
    model.closeAllConnections()
    await new Promise<void>(resolve => model.close(() => resolve()))
    fs.rmSync(dir, { recursive: true, force: true })
  }
}, 60000)
