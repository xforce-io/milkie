/**
 * #85: confirm interrupt/resume works across the HTTP boundary via a sidecar.
 *
 * Architecture: alfred (an external process) cannot embed the Node milkie SDK,
 * so it drives a milkie sidecar over HTTP. The cross-process boundary IS the
 * HTTP layer — inside the sidecar, milkie.interrupt()/resume() are in-process.
 *
 * Level 1 (this file): a real http.Server bound to an ephemeral port, driven by
 * real `fetch` over TCP. Proves the HTTP endpoints correctly wire to the SDK
 * while a run is in flight. Level 2 (separate describe) spawns the sidecar as a
 * distinct OS process for the full cross-process guarantee.
 */
import type { Server } from 'http'
import { createSidecar, buildDemoMilkie } from '../../examples/interrupt-resume-sidecar/sidecar.js'

function listen(server: Server): Promise<number> {
  return new Promise(resolve => server.listen(0, () => {
    const addr = server.address()
    resolve(typeof addr === 'object' && addr ? addr.port : 0)
  }))
}
function close(server: Server): Promise<void> {
  return new Promise(resolve => server.close(() => resolve()))
}

async function waitFor(predicate: () => Promise<boolean>, timeoutMs = 8000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (await predicate()) return
    await new Promise<void>(r => setTimeout(r, 25))
  }
  throw new Error('timed out waiting for condition')
}

describe('#85 interrupt/resume over HTTP sidecar (in-process)', () => {
  let server: Server
  let base: string
  let demo: ReturnType<typeof buildDemoMilkie>

  beforeEach(async () => {
    demo = buildDemoMilkie()
    server = createSidecar({ milkie: demo.milkie, agentId: demo.agentId, progress: () => demo.executedSteps().length })
    const port = await listen(server)
    base = `http://127.0.0.1:${port}`
  })
  afterEach(async () => { await close(server) })

  const post = (path: string, body: unknown) =>
    fetch(base + path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const getJson = async (path: string): Promise<any> => (await fetch(base + path)).json()

  it('health responds ok', async () => {
    const res = await fetch(base + '/health')
    expect(res.status).toBe(200)
    expect(await res.json()).toMatchObject({ ok: true })
  })

  it('external HTTP interrupt stops an in-flight run; resume continues from the breakpoint', async () => {
    const contextId = 'c-http-1'

    // start a long run; /chat returns immediately (run continues async)
    const chat = await post('/chat', { contextId, goal: 'work', input: 'start' })
    expect(chat.status).toBe(202)

    // wait until the run is observably mid-flight
    await waitFor(async () => (await getJson(`/status?contextId=${contextId}`)).steps >= 2)

    // external interrupt over HTTP
    const intr = await post('/interrupt', { contextId })
    expect(intr.status).toBe(200)
    expect(await intr.json()).toMatchObject({ signaled: true })

    // the in-flight run settles as interrupted
    await waitFor(async () => (await getJson(`/status?contextId=${contextId}`)).state === 'interrupted')
    const atInterrupt = (await getJson(`/status?contextId=${contextId}`)).steps as number
    expect(atInterrupt).toBeGreaterThanOrEqual(2)
    expect(atInterrupt).toBeLessThan(demo.totalSteps)   // did NOT finish

    // resume with new input → run completes
    const resumeRes = await post('/resume', { contextId, input: 'continue' })
    expect(resumeRes.status).toBe(200)
    expect(await resumeRes.json()).toMatchObject({ status: 'completed' })

    // every step ran exactly once, all the way through — proving continuation,
    // not a restart from step 1.
    const executed = demo.executedSteps()
    expect(executed).toEqual(Array.from({ length: demo.totalSteps }, (_, i) => i + 1))
  })
})
