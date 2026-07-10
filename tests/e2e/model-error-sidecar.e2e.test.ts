import { spawn, spawnSync, type ChildProcess } from 'child_process'
import fs from 'fs'
import http from 'http'
import os from 'os'
import path from 'path'

const REPO_ROOT = path.resolve(__dirname, '..', '..')
const CLI = path.join('dist', 'cli', 'index.js')

interface SSEEvent { event: string; data: unknown }

function parseSse(text: string): SSEEvent[] {
  return text.split('\n\n').flatMap(raw => {
    let event = 'message'
    const data: string[] = []
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim()
      if (line.startsWith('data:')) data.push(line.slice(5).trim())
    }
    return data.length > 0 ? [{ event, data: JSON.parse(data.join('\n')) as unknown }] : []
  })
}

function startServe(agentFile: string, dataDir: string): Promise<{ child: ChildProcess; port: number; output: () => string }> {
  return new Promise((resolve, reject) => {
    const proc = spawn(process.execPath, [
      CLI, 'serve', '--agent', agentFile, '--port', '0',
      '--state-store', 'sqlite', '--data-dir', dataDir,
    ], {
      cwd: REPO_ROOT,
      env: { ...process.env, VOLCENGINE_TOKEN: 'test-token', LOG_FORMAT: 'json' },
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    let output = ''
    const timer = setTimeout(() => {
      cleanup()
      proc.kill('SIGKILL')
      reject(new Error(`serve did not become ready:\n${output}`))
    }, 30000)
    const onExit = (code: number | null) => {
      cleanup()
      reject(new Error(`serve exited before ready (${code}):\n${output}`))
    }
    const onData = (chunk: Buffer) => {
      output += chunk.toString()
      const match = output.match(/MILKIE_SERVE_READY (\d+)/)
      if (match) {
        cleanup()
        resolve({ child: proc, port: Number(match[1]), output: () => output })
      }
    }
    const collect = (chunk: Buffer) => { output += chunk.toString() }
    function cleanup(): void {
      clearTimeout(timer)
      proc.stdout?.off('data', onData)
      proc.off('exit', onExit)
    }
    proc.stdout?.on('data', onData)
    proc.stderr?.on('data', collect)
    proc.on('exit', onExit)
  })
}

function stop(proc: ChildProcess): Promise<void> {
  return new Promise(resolve => {
    if (proc.exitCode !== null || proc.signalCode !== null) return resolve()
    proc.once('exit', () => resolve())
    proc.kill('SIGTERM')
    setTimeout(() => proc.kill('SIGKILL'), 4000).unref()
  })
}

describe('#202 structured model errors — real serve subprocess', () => {
  let provider: http.Server
  let providerPort: number
  let child: ChildProcess | undefined
  let tempDir: string

  beforeAll(async () => {
    provider = http.createServer((req) => {
      req.resume()
      req.socket.destroy()
    })
    providerPort = await new Promise<number>(resolve => {
      provider.listen(0, '127.0.0.1', () => {
        resolve((provider.address() as { port: number }).port)
      })
    })
  })

  afterAll(async () => {
    if (child) await stop(child)
    await new Promise<void>(resolve => provider.close(() => resolve()))
    if (tempDir && process.env['KEEP_E2E_ARTIFACTS'] !== '1') {
      fs.rmSync(tempDir, { recursive: true, force: true })
    }
  })

  it('persists and renders a retryable connection envelope while serve remains healthy', async () => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-model-error-'))
    const agentFile = path.join(tempDir, 'agent.md')
    fs.writeFileSync(agentFile, `---
agentId: failing-agent
version: 1.0.0
fsm:
  states:
    - name: react
      type: llm
model:
  provider: local-stub
  model: failing-model
  adapter: openai-compatible
  baseUrl: http://127.0.0.1:${providerPort}
---
fail deterministically
`)

    const started = await startServe(agentFile, tempDir)
    child = started.child
    const base = `http://127.0.0.1:${started.port}`
    const chat = await fetch(base + '/chat', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ contextId: 'model-error-e2e', input: 'go' }),
    })
    const events = parseSse(await chat.text())
    const error = events.find(event => event.event === 'error')
    const terminal = events.find(event => event.event === 'agent.run.completed')

    expect(error?.data).toMatchObject({
      error: {
        code: 'MODEL_CONNECTION_ERROR', phase: 'stream_open', provider: 'local-stub',
        model: 'failing-model', retryable: true,
      },
    })
    expect(terminal?.data).toMatchObject({ status: 'error', runId: expect.any(String) })
    const runId = (terminal!.data as { runId: string }).runId

    const health = await fetch(base + '/health')
    expect(await health.json()).toEqual({ ok: true })

    const traceFile = path.join(tempDir, 'runs', `${runId}.jsonl`)
    const trace = fs.readFileSync(traceFile, 'utf8')
    expect(trace).toContain('MODEL_CONNECTION_ERROR')
    expect(trace).not.toContain('test-token')

    const report = spawnSync(process.execPath, [CLI, 'trace', 'report', '--data-dir', tempDir, runId], {
      cwd: REPO_ROOT, encoding: 'utf8', timeout: 30000,
    })
    if (process.env['KEEP_E2E_ARTIFACTS'] === '1') process.stderr.write(`E2E artifacts: ${tempDir}\n`)
    expect(report.status).toBe(0)
    expect(report.stdout).toContain(runId)
    expect(report.stdout).toContain('badge error')

    await stop(started.child)
    child = undefined
  }, 45000)
})
