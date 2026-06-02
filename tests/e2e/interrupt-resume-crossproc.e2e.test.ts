/**
 * #85 Level 2: TRUE cross-process interrupt/resume over HTTP.
 *
 * The sidecar runs as a separate OS process (spawned via `npx tsx main.ts`).
 * This test process plays the role of alfred: it knows nothing of the sidecar's
 * internals and drives it purely over HTTP — start a long run, interrupt it
 * mid-flight, then resume it with new input. This is the end-to-end proof for
 * issue #85's acceptance criterion.
 */
import { spawn, type ChildProcess } from 'child_process'
import path from 'path'

const REPO_ROOT = path.resolve(__dirname, '..', '..')
const MAIN = path.join('examples', 'interrupt-resume-sidecar', 'main.ts')
const TOTAL_STEPS = 8

let child: ChildProcess
let base: string

function startSidecar(): Promise<{ child: ChildProcess; port: number }> {
  return new Promise((resolve, reject) => {
    const proc = spawn('npx', ['tsx', MAIN], {
      cwd: REPO_ROOT,
      env: { ...process.env, PORT: '0', STEPS: String(TOTAL_STEPS), STEP_MS: '200' },
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let out = ''
    const timer = setTimeout(() => { cleanup(); reject(new Error(`sidecar did not become ready in time; output:\n${out}`)) }, 30000)
    const onExit = (code: number | null) => { cleanup(); reject(new Error(`sidecar exited early (code ${code}); output:\n${out}`)) }
    const onData = (buf: Buffer) => {
      out += buf.toString()
      const m = out.match(/SIDECAR_READY (\d+)/)
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

function stopSidecar(proc: ChildProcess): Promise<void> {
  return new Promise(resolve => {
    if (proc.exitCode !== null || proc.signalCode !== null) return resolve()
    proc.on('exit', () => resolve())
    proc.kill('SIGTERM')
    setTimeout(() => { proc.kill('SIGKILL'); resolve() }, 3000).unref()
  })
}

async function waitFor(predicate: () => Promise<boolean>, timeoutMs = 15000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (await predicate()) return
    await new Promise<void>(r => setTimeout(r, 40))
  }
  throw new Error('timed out waiting for condition')
}

beforeAll(async () => {
  const started = await startSidecar()
  child = started.child
  base = `http://127.0.0.1:${started.port}`
}, 35000)

afterAll(async () => { if (child) await stopSidecar(child) })

const post = (p: string, body: unknown) =>
  fetch(base + p, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const getJson = async (p: string): Promise<any> => (await fetch(base + p)).json()

describe('#85 interrupt/resume across a separate OS process (HTTP)', () => {
  it('the sidecar process is reachable over HTTP', async () => {
    const res = await fetch(base + '/health')
    expect(res.status).toBe(200)
    expect(await res.json()).toMatchObject({ ok: true })
  })

  it('external process interrupts an in-flight run, then resumes it to completion', async () => {
    const contextId = 'xproc-1'

    const chat = await post('/chat', { contextId, goal: 'work', input: 'start' })
    expect(chat.status).toBe(202)

    // run is observably mid-flight in the OTHER process
    await waitFor(async () => (await getJson(`/status?contextId=${contextId}`)).steps >= 2)

    const intr = await post('/interrupt', { contextId })
    expect(intr.status).toBe(200)
    expect(await intr.json()).toMatchObject({ signaled: true })

    // the run in the other process settles as interrupted, before finishing
    await waitFor(async () => (await getJson(`/status?contextId=${contextId}`)).state === 'interrupted')
    const atInterrupt = (await getJson(`/status?contextId=${contextId}`)).steps as number
    expect(atInterrupt).toBeGreaterThanOrEqual(2)
    expect(atInterrupt).toBeLessThan(TOTAL_STEPS)

    // resume with new input → completes
    const resumeRes = await post('/resume', { contextId, input: 'continue' })
    expect(resumeRes.status).toBe(200)
    expect(await resumeRes.json()).toMatchObject({ status: 'completed' })

    // all steps eventually ran (continuation across the interrupt, in the
    // sidecar process); /status reports the full count
    await waitFor(async () => (await getJson(`/status?contextId=${contextId}`)).steps === TOTAL_STEPS)
  })
})
