// Repair-ticketing playground server (#162 / S-011 Path D as a UI).
//
// Plain node `http` — no framework — matching the agent-docs-qa example. Three
// routes:
//   GET  /                     → serve public/index.html
//   POST /chat                 → run one milkie turn; return status/output, the
//                                reconstructed workingMemory, and the assembled
//                                ticket (parsed from the live invoke output).
//   GET  /events/:contextId    → SSE stream of WM-change events (wm.mutated) so
//                                the UI fills slot chips in real time.
//
// The model is the agent's own (volcengine doubao via openai-compatible); set the
// provider credentials in the environment before running. Tests inject a
// deterministic gateway instead — see src/__tests__/repair-ticketing.e2e.test.ts.

import http, { type IncomingMessage, type ServerResponse, type Server } from 'http'
import { promises as fs, readFileSync } from 'fs'
import path from 'path'
import { v4 as uuidv4 } from 'uuid'

import { Milkie } from '../../../src/runtime/Milkie.js'
import { MemoryStore } from '../../../src/store/MemoryStore.js'
import { MemoryEventStore } from '../../../src/trace/MemoryEventStore.js'
import { BroadcastingEventStore } from '../../../src/trace/BroadcastingEventStore.js'
import { OpenAICompatibleAdapter } from '../../../src/gateway/OpenAICompatibleAdapter.js'
import type { IModelGateway } from '../../../src/types/model.js'
import type { Event } from '../../../src/trace/types.js'

import { EntityResolver, type Schema } from '../resolver/EntityResolver.js'
import { buildRepairTicketingTools, repairTicketingAgentConfig, type Ticket } from './agent.js'

export interface ServerConfig {
  port:        number
  exampleDir:  string
  /** Test injection. When omitted Milkie falls back to the agent's own model. */
  gateway?:    IModelGateway
}

interface ServerState {
  milkie:     Milkie
  eventStore: BroadcastingEventStore
  publicDir:  string
  /** All runIds seen per contextId, oldest-first — for SSE catch-up across turns. */
  runsByContext: Map<string, string[]>
}

async function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    req.on('data', c => chunks.push(c as Buffer))
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')))
    req.on('error', reject)
  })
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { 'content-type': 'application/json' })
  res.end(JSON.stringify(body))
}

/** A `wm.mutated` payload carries the full WorkingMemory as `{ data, log }`;
 *  the UI only wants the flat field map under `data`. */
function flatData(snapshot: unknown): Record<string, unknown> {
  return (snapshot as { data?: Record<string, unknown> } | null)?.data ?? {}
}

/** The cumulative WM field map for a run = its last `wm.mutated` snapshot (each
 *  snapshot is the full WorkingMemory, not a delta). Empty object if the run made
 *  no tool-driven WM writes. */
function latestWorkingMemory(events: Event[]): Record<string, unknown> {
  let data: Record<string, unknown> = {}
  for (const e of events) {
    if (e.type === 'wm.mutated') {
      data = flatData((e.payload as { snapshot: unknown }).snapshot)
    }
  }
  return data
}

/** True when a value is an assembled ticket object (has a string ticketId). */
function isTicket(v: unknown): v is Ticket {
  return !!v && typeof v === 'object' && typeof (v as Partial<Ticket>).ticketId === 'string'
}

/** Parse the live invoke output as a ticket, or null when this turn produced
 *  plain text (i.e. assemble_ticket did not run / its precondition was unmet). */
function parseTicket(output: string): Ticket | null {
  try {
    const parsed = JSON.parse(output) as Partial<Ticket>
    return isTicket(parsed) ? parsed : null
  } catch {
    return null
  }
}

async function handleChat(req: IncomingMessage, res: ServerResponse, s: ServerState): Promise<void> {
  const { input, contextId } = JSON.parse(await readBody(req)) as { input: string; contextId?: string }
  if (!input) { sendJson(res, 400, { error: 'input required' }); return }

  const ctxId  = contextId ?? uuidv4()
  const result = await s.milkie.invoke({
    agentId:   repairTicketingAgentConfig.agentId,
    goal:      '登记一张报修工单',
    input,
    contextId: ctxId,
  })

  const runs = s.runsByContext.get(ctxId) ?? []
  runs.push(result.agentRunId)
  s.runsByContext.set(ctxId, runs)

  const events = await s.eventStore.readByRunId(result.agentRunId)
  const wm = latestWorkingMemory(events)
  // Prefer the ticket object assemble_ticket stored in WM (authoritative,
  // model-paraphrase-proof — same source run-eval uses); fall back to parsing the
  // output only for older shapes. The model often reformats the ticket JSON into
  // markdown prose, which JSON.parse can't read — reading WM keeps the ticket CARD
  // rendering instead of leaking a raw markdown blob into the chat.
  sendJson(res, 200, {
    contextId:     ctxId,
    runId:         result.agentRunId,
    status:        result.status,
    output:        result.output,
    workingMemory: wm,
    ticket:        isTicket(wm['ticket']) ? wm['ticket'] : parseTicket(result.output),
  })
}

async function handleSse(req: IncomingMessage, res: ServerResponse, s: ServerState, contextId: string): Promise<void> {
  res.writeHead(200, {
    'content-type':      'text/event-stream',
    'cache-control':     'no-cache',
    'connection':        'keep-alive',
    'x-accel-buffering': 'no',
  })

  const write = (event: Event): void => {
    if (event.type === 'wm.mutated' && !res.writableEnded) {
      const snapshot = flatData((event.payload as { snapshot: unknown }).snapshot)
      res.write(`data: ${JSON.stringify({ type: 'wm', snapshot })}\n\n`)
    }
  }

  // Catch-up: replay past WM-change events across every turn of this context.
  for (const runId of s.runsByContext.get(contextId) ?? []) {
    for (const e of await s.eventStore.readByRunId(runId)) write(e)
  }

  const unsubscribe = s.eventStore.subscribe(contextId, write)
  req.on('close', () => {
    unsubscribe()
    if (!res.writableEnded) res.end()
  })
}

async function serveIndex(res: ServerResponse, publicDir: string): Promise<void> {
  try {
    const html = await fs.readFile(path.join(publicDir, 'index.html'), 'utf-8')
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
    res.end(html)
  } catch {
    res.writeHead(404).end()
  }
}

export async function startServer(config: ServerConfig): Promise<Server> {
  const resolverDir = path.join(config.exampleDir, 'resolver')
  const schema = JSON.parse(readFileSync(path.join(resolverDir, 'schema.json'), 'utf-8')) as Schema
  const csv    = readFileSync(path.join(resolverDir, 'data.csv'), 'utf-8')
  const resolver = EntityResolver.load(schema, csv)

  // Eval-parity model override (mirrors eval/run-eval.ts): the committed example
  // ships doubao-seed-2.0-lite, too weak to drive the flow well. When DEEPSEEK_API_KEY
  // is set (and no test gateway is injected), run the SAME agent against DeepSeek so
  // the playground matches the eval. The committed config/default is untouched.
  const useDeepseek = !config.gateway && !!process.env['DEEPSEEK_API_KEY']
  const agentConfig = useDeepseek
    ? {
        ...repairTicketingAgentConfig,
        model: {
          provider: 'deepseek',
          model:    process.env['EVAL_MODEL'] ?? 'deepseek-chat',
          adapter:  'openai-compatible' as const,
          baseUrl:  process.env['DEEPSEEK_API_BASE'] ?? 'https://api.deepseek.com',
        },
      }
    : repairTicketingAgentConfig

  const eventStore = new BroadcastingEventStore(new MemoryEventStore())
  const milkie = new Milkie({
    stateStore: new MemoryStore(),
    eventStore,
    // test gateway → use it; else DeepSeek override when configured; else omitted
    // (Milkie builds the agent's own model gateway from agentConfig.model).
    gateway:    config.gateway
      ?? (useDeepseek
        ? new OpenAICompatibleAdapter({
            baseUrl: process.env['DEEPSEEK_API_BASE'] ?? 'https://api.deepseek.com',
            apiKey:  process.env['DEEPSEEK_API_KEY'],
          })
        : undefined),
  })
  for (const tool of buildRepairTicketingTools(resolver)) milkie.registerTool(tool)
  milkie.registerAgent(agentConfig)

  const state: ServerState = {
    milkie,
    eventStore,
    publicDir:     path.join(config.exampleDir, 'public'),
    runsByContext: new Map(),
  }

  const server = http.createServer(async (req, res) => {
    try {
      const route = new URL(req.url ?? '/', 'http://localhost').pathname

      if (req.method === 'POST' && route === '/chat') return handleChat(req, res, state)

      const sse = route.match(/^\/events\/([^/]+)$/)
      if (req.method === 'GET' && sse) return handleSse(req, res, state, decodeURIComponent(sse[1]!))

      if (req.method === 'GET' && (route === '/' || route === '/index.html')) {
        return serveIndex(res, state.publicDir)
      }
      res.writeHead(404).end()
    } catch (err) {
      sendJson(res, 500, { error: (err as Error).message })
    }
  })

  await new Promise<void>(resolve => server.listen(config.port, () => resolve()))
  return server
}

export async function stopServer(server: Server): Promise<void> {
  await new Promise<void>(resolve => server.close(() => resolve()))
}

// CLI entry — only when run directly (`npx tsx src/server.ts`). Tests import
// startServer/stopServer and inject their own gateway. __dirname keeps this
// compatible with both ts-jest (CommonJS) and tsx.
const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(__filename)
if (isMain) {
  const PORT = Number(process.env.PORT ?? 7979)
  startServer({ port: PORT, exampleDir: path.dirname(__dirname) })
    .then(() => console.log(`repair-ticketing playground at http://localhost:${PORT}`))
}
