/**
 * #85: a minimal, deterministic milkie sidecar demonstrating cross-process
 * interrupt/resume over HTTP.
 *
 * alfred (a Python process) cannot embed the Node milkie SDK, so it drives a
 * sidecar over HTTP. The cross-process boundary is the HTTP layer; inside the
 * sidecar, `milkie.interrupt()` / `milkie.resume()` are ordinary in-process SDK
 * calls. Because the sidecar is a single Node process, an interrupt request that
 * arrives while a run is in flight is serviced on the same event loop and the
 * running AgentRuntime observes the signal at its next yield point.
 *
 * Endpoints:
 *   GET  /health                 → { ok: true }
 *   POST /chat    {contextId,goal,input}  → 202; starts a run, returns immediately
 *   GET  /status?contextId=...   → { state, steps }
 *   POST /interrupt {contextId}  → { signaled: true }
 *   POST /resume  {contextId,input}       → { status, output }
 *
 * `sidecar.ts` is side-effect free (only exports). The standalone launcher lives
 * in `main.ts` so this module can be imported by tests without starting a server.
 */
import http, { type IncomingMessage, type ServerResponse, type Server } from 'http'
import { Milkie } from '../../src/runtime/Milkie.js'
import { MemoryStore } from '../../src/store/MemoryStore.js'
import { MemoryEventStore } from '../../src/trace/MemoryEventStore.js'
import type { AgentConfig } from '../../src/types/agent.js'
import type { AgentResult } from '../../src/types/common.js'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../src/types/model.js'
import type { ToolDefinition } from '../../src/types/tool.js'

// ─────────────────────────── deterministic demo agent ────────────────────────

export interface DemoMilkie {
  milkie:       Milkie
  agentId:      string
  totalSteps:   number
  /** Step ids executed so far, sorted — used by tests to assert continuation. */
  executedSteps: () => number[]
}

/**
 * Build a Milkie whose single agent loops `totalSteps` times: each LLM turn the
 * baked-in gateway issues one `work_step` tool call (which sleeps briefly so an
 * interrupt can land between calls), then finishes once all steps are done. The
 * gateway counter and the executed-step set live in this process, so a run that
 * is interrupted and later resumed continues from where it stopped.
 */
export function buildDemoMilkie(opts: { totalSteps?: number; stepMs?: number } = {}): DemoMilkie {
  const totalSteps = opts.totalSteps ?? 8
  const stepMs     = opts.stepMs ?? 150
  const executed   = new Set<number>()
  let counter = 0

  const workStep: ToolDefinition = {
    name:        'work_step',
    description: 'Perform one unit of work.',
    inputSchema: { type: 'object', properties: { stepId: { type: 'number' } }, required: ['stepId'] },
    handler: async (input: unknown) => {
      const { stepId } = input as { stepId: number }
      await new Promise<void>(r => setTimeout(r, stepMs))
      executed.add(stepId)
      return { stepId, done: true }
    },
  }

  const gateway: IModelGateway = {
    async complete(_req: ModelRequest): Promise<ModelResponse> {
      if (counter >= totalSteps) {
        return { content: [{ type: 'text', text: `done: ${[...executed].sort((a, b) => a - b).join(',')}` }], toolCalls: [], finishReason: 'end_turn' }
      }
      counter++
      const stepId = counter
      const input = { stepId }
      return { content: [{ type: 'tool_use', id: `w${stepId}`, name: 'work_step', input }], toolCalls: [{ id: `w${stepId}`, name: 'work_step', input }], finishReason: 'tool_use' }
    },
    async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] },
  }

  const agent: AgentConfig = {
    agentId:      'stepper',
    version:      '1.0.0',
    systemPrompt: 'Call work_step once per turn until done.',
    fsm:          { states: [{ name: 'react', type: 'llm', max_iterations: totalSteps + 5 }] },
    model:        { provider: 'stub', model: 'stub', adapter: 'stub' },
  }

  const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore(), gateway, tools: [workStep] })
  milkie.registerAgent(agent)

  return { milkie, agentId: agent.agentId, totalSteps, executedSteps: () => [...executed].sort((a, b) => a - b) }
}

// ───────────────────────────────── HTTP sidecar ──────────────────────────────

type RunState = 'running' | 'interrupted' | 'completed' | 'error'

export interface SidecarOptions {
  milkie:  Milkie
  agentId: string
  /** Optional probe reporting completed work units (for /status progress). */
  progress?: () => number
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = ''
    req.on('data', c => { data += c })
    req.on('end', () => resolve(data))
    req.on('error', reject)
  })
}
function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body)
  res.writeHead(status, { 'content-type': 'application/json' })
  res.end(payload)
}

/**
 * Create (but do not start) the sidecar HTTP server. Caller listens on a port.
 */
export function createSidecar(opts: SidecarOptions): Server {
  const { milkie, agentId, progress } = opts

  // Per-context run state. Progress (completed work units) comes from the probe.
  const runs = new Map<string, { state: RunState; output?: string }>()
  const steps = () => (progress ? progress() : 0)

  async function handleChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const { contextId, goal, input } = JSON.parse(await readBody(req)) as { contextId: string; goal?: string; input?: string }
    runs.set(contextId, { state: 'running' })
    // Fire-and-forget: the run continues on the event loop; /interrupt can arrive
    // concurrently and is serviced on the same loop while the run awaits I/O.
    void milkie.invoke({ agentId, goal: goal ?? 'work', input: input ?? 'start', contextId }).then(
      (result: AgentResult) => { runs.set(contextId, { state: result.status === 'completed' ? 'completed' : result.status === 'interrupted' ? 'interrupted' : 'error', output: result.output }) },
      (err: unknown)        => { runs.set(contextId, { state: 'error', output: String(err) }) },
    )
    sendJson(res, 202, { contextId, accepted: true })
  }

  function handleStatus(res: ServerResponse, contextId: string): void {
    const r = runs.get(contextId)
    sendJson(res, 200, r ? { state: r.state, steps: steps(), output: r.output } : { state: 'unknown', steps: steps() })
  }

  async function handleInterrupt(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const { contextId } = JSON.parse(await readBody(req)) as { contextId: string }
    await milkie.interrupt(contextId)
    sendJson(res, 200, { contextId, signaled: true })
  }

  async function handleResume(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const { contextId, input } = JSON.parse(await readBody(req)) as { contextId: string; input?: string }
    runs.set(contextId, { state: 'running' })
    try {
      const result = await milkie.resume(
        `context:${contextId}:checkpoint:latest`,
        agentId,
        'continue',
        input ?? 'continue',
      )
      runs.set(contextId, { state: result.status === 'completed' ? 'completed' : result.status === 'interrupted' ? 'interrupted' : 'error', output: result.output })
      sendJson(res, 200, { contextId, status: result.status, output: result.output })
    } catch (err) {
      runs.set(contextId, { state: 'error' })
      sendJson(res, 400, { error: err instanceof Error ? err.message : String(err) })
    }
  }

  return http.createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? '/', 'http://localhost')
        const route = url.pathname
        if (req.method === 'GET' && route === '/health') return sendJson(res, 200, { ok: true })
        if (req.method === 'POST' && route === '/chat') return await handleChat(req, res)
        if (req.method === 'GET' && route === '/status') return handleStatus(res, url.searchParams.get('contextId') ?? '')
        if (req.method === 'POST' && route === '/interrupt') return await handleInterrupt(req, res)
        if (req.method === 'POST' && route === '/resume') return await handleResume(req, res)
        sendJson(res, 404, { error: `no route ${req.method} ${route}` })
      } catch (err) {
        sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) })
      }
    })()
  })
}
