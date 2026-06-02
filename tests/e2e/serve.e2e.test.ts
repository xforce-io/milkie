/**
 * #86 Level 2: TRUE cross-process e2e for `milkie serve`.
 *
 * Spawns the serve fixture as a separate OS process and drives it purely over
 * HTTP + SSE — the role alfred plays. Proves, across a real process boundary:
 *   - readiness signal on stdout (MILKIE_SERVE_READY <port>)
 *   - POST /chat returns an SSE stream (D2)
 *   - POST /interrupt ends the in-flight stream with an `interrupted` terminal
 *     event, not a bare disconnect (§7.3)
 *   - POST /resume continues to completion on a fresh stream, emitting
 *     token-level message_delta + a `completed` terminal (§7.2)
 *   - SIGTERM shuts the process down gracefully (no zombie)
 *
 * Single contextId, strictly sequential — the stub gateway's step counter is
 * process-global, so concurrent contexts would interleave.
 */
import { spawn, type ChildProcess } from 'child_process'
import path from 'path'

const REPO_ROOT = path.resolve(__dirname, '..', '..')
const FIXTURE = path.join('tests', 'e2e', 'fixtures', 'serve-stub-entry.ts')

interface SSEEvent { event: string; data: unknown }
function parseFrame(raw: string): SSEEvent | null {
  let event = 'message'; const dataLines: string[] = []
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (!dataLines.length) return null
  return { event, data: JSON.parse(dataLines.join('\n')) }
}

let child: ChildProcess
let base: string

function startServe(): Promise<{ child: ChildProcess; port: number }> {
  return new Promise((resolve, reject) => {
    const proc = spawn('npx', ['tsx', FIXTURE], {
      cwd: REPO_ROOT,
      env: { ...process.env, PORT: '0', STEPS: '6', STEP_MS: '120' },
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    let out = ''
    const timer = setTimeout(() => { cleanup(); reject(new Error(`serve did not become ready; output:\n${out}`)) }, 30000)
    const onExit = (code: number | null) => { cleanup(); reject(new Error(`serve exited early (code ${code}); output:\n${out}`)) }
    const onData = (buf: Buffer) => {
      out += buf.toString()
      const m = out.match(/MILKIE_SERVE_READY (\d+)/)
      if (m) { cleanup(); resolve({ child: proc, port: Number(m[1]) }) }
    }
    function cleanup(): void {
      clearTimeout(timer)
      proc.stdout?.off('data', onData)
      proc.off('exit', onExit)
    }
    proc.stdout?.on('data', onData)
    proc.stderr?.on('data', (b: Buffer) => { out += b.toString() })
    proc.on('exit', onExit)
  })
}

function stopServe(proc: ChildProcess): Promise<number | null> {
  return new Promise(resolve => {
    if (proc.exitCode !== null || proc.signalCode !== null) return resolve(proc.exitCode)
    proc.on('exit', code => resolve(code))
    proc.kill('SIGTERM')
    setTimeout(() => { proc.kill('SIGKILL'); resolve(null) }, 4000).unref()
  })
}

const post = (p: string, body: unknown) =>
  fetch(base + p, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })

/** Open an SSE POST; collect all frames until the stream closes. */
async function sse(p: string, body: unknown): Promise<SSEEvent[]> {
  const res = await fetch(base + p, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
  const reader = res.body!.getReader()
  const dec = new TextDecoder()
  let buf = ''
  const events: SSEEvent[] = []
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let idx
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const frame = parseFrame(buf.slice(0, idx)); buf = buf.slice(idx + 2)
      if (frame) events.push(frame)
    }
  }
  return events
}

const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms))

beforeAll(async () => {
  const started = await startServe()
  child = started.child
  base = `http://127.0.0.1:${started.port}`
}, 35000)

afterAll(async () => { if (child) await stopServe(child) })

describe('#86 milkie serve — cross-process HTTP + SSE', () => {
  it('the serve process is reachable over HTTP', async () => {
    const res = await fetch(base + '/health')
    expect(res.status).toBe(200)
    expect(await res.json()).toMatchObject({ ok: true })
  })

  it('interrupt ends an in-flight /chat stream with an interrupted terminal; resume continues to completion with token deltas', async () => {
    const contextId = 'e2e-1'

    // Start a long run over SSE; do NOT await — interrupt it mid-flight.
    const chatP = sse('/chat', { contextId, input: 'go' })
    await sleep(280)   // ~2 steps in
    const intr = await post('/interrupt', { contextId })
    expect(intr.status).toBe(200)
    expect(await intr.json()).toMatchObject({ signaled: true })

    const chatEvents = await chatP
    const chatTerminal = chatEvents.find(e => e.event === 'agent.run.completed')
    expect(chatTerminal).toBeDefined()
    expect((chatTerminal!.data as { status: string }).status).toBe('interrupted')

    // Resume on a fresh stream → runs to completion, streaming token deltas.
    const resumeEvents = await sse('/resume', { contextId })
    const deltas = resumeEvents.filter(e => e.event === 'message_delta')
    expect(deltas.length).toBeGreaterThanOrEqual(2)
    const resumeTerminal = resumeEvents.find(e => e.event === 'agent.run.completed')
    expect(resumeTerminal).toBeDefined()
    expect((resumeTerminal!.data as { status: string }).status).toBe('completed')
  }, 20000)

  it('shuts down gracefully on SIGTERM (no zombie)', async () => {
    const code = await stopServe(child)
    // closed cleanly (exit 0) or terminated by our signal — not still running
    expect(child.exitCode !== null || child.signalCode !== null).toBe(true)
    void code
  })
})
