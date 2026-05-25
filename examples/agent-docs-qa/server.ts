import http, { type IncomingMessage, type ServerResponse, type Server } from 'http'
import { promises as fs } from 'fs'
import { existsSync, mkdirSync } from 'fs'
import path from 'path'
import { v4 as uuidv4 } from 'uuid'
import { Milkie } from '../../src/runtime/Milkie.js'
import { MemoryStore } from '../../src/store/MemoryStore.js'
import { JsonlEventStore } from '../../src/trace/JsonlEventStore.js'
import type { IModelGateway } from '../../src/types/model.js'
import { BroadcastingEventStore } from './trace/broadcast-event-store.js'
import { scanConversations, readEventsForContext } from './trace/conversation-scanner.js'
import { makeCorpusToolDefinitions } from './tools/corpus-tools.js'

export interface ServerConfig {
  port:        number
  exampleDir:  string
  gateway?:    IModelGateway      // override (test injection); when absent, Milkie falls back to createGateway(config.model)
  agentFile:   string
  corpusRoot:  string
}

interface ServerState {
  milkie:      Milkie
  eventStore:  BroadcastingEventStore
  runsDir:     string
  publicDir:   string
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
  const json = JSON.stringify(body)
  res.writeHead(status, { 'content-type': 'application/json' })
  res.end(json)
}

async function handleChat(req: IncomingMessage, res: ServerResponse, s: ServerState): Promise<void> {
  const raw = await readBody(req)
  const { input, contextId } = JSON.parse(raw) as { input: string; contextId?: string }
  if (!input) { sendJson(res, 400, { error: 'input required' }); return }

  const ctxId = contextId ?? uuidv4()
  const result = await s.milkie.invoke({
    agentId:   'sanguo-researcher',
    goal:      input,
    input,
    contextId: ctxId,
  })

  sendJson(res, 200, {
    runId:     result.agentRunId,
    contextId: ctxId,
    status:    result.status,
    output:    result.output,
  })
}

async function handleListConversations(res: ServerResponse, s: ServerState): Promise<void> {
  const conversations = await scanConversations(s.runsDir)
  sendJson(res, 200, { conversations })
}

async function handleGetConversationEvents(
  res: ServerResponse, s: ServerState, contextId: string,
): Promise<void> {
  const events = await readEventsForContext(s.runsDir, contextId)
  if (events.length === 0) { sendJson(res, 404, { error: 'conversation not found' }); return }
  sendJson(res, 200, { events })
}

async function serveStatic(res: ServerResponse, filePath: string): Promise<void> {
  try {
    const content = await fs.readFile(filePath, 'utf-8')
    const ext = path.extname(filePath)
    const ctype = ext === '.html' ? 'text/html; charset=utf-8' : 'text/plain'
    res.writeHead(200, { 'content-type': ctype })
    res.end(content)
  } catch {
    res.writeHead(404).end()
  }
}

export async function startServer(config: ServerConfig): Promise<Server> {
  const runsDir = path.join(config.exampleDir, '.milkie', 'runs')
  if (!existsSync(runsDir)) mkdirSync(runsDir, { recursive: true })

  const eventStore = new BroadcastingEventStore(new JsonlEventStore(runsDir))
  const milkie     = new Milkie({
    stateStore: new MemoryStore(),
    gateway:    config.gateway,   // when omitted, Milkie falls back to createGateway(agent.model) per-invoke
    eventStore,
  })

  for (const tool of makeCorpusToolDefinitions(config.corpusRoot)) {
    milkie.registerTool(tool)
  }
  milkie.loadAgentFile(config.agentFile)

  // Public dir resolution: prefer co-located public/ when present (production
  // mode using the real example dir), else fall back to this file's
  // sibling public/ (test mode where exampleDir is a tmpDir).
  const colocatedPublic = path.join(config.exampleDir, 'public')
  const fallbackPublic  = path.resolve(__dirname, 'public')
  const publicDir = existsSync(colocatedPublic) ? colocatedPublic : fallbackPublic

  const state: ServerState = { milkie, eventStore, runsDir, publicDir }

  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', 'http://localhost')
      const route = url.pathname

      if (req.method === 'POST' && route === '/chat')
        return handleChat(req, res, state)
      if (req.method === 'GET' && route === '/conversations')
        return handleListConversations(res, state)

      const convMatch = route.match(/^\/conversation\/([^/]+)\/events$/)
      if (req.method === 'GET' && convMatch) {
        return handleGetConversationEvents(res, state, decodeURIComponent(convMatch[1]!))
      }

      if (req.method === 'GET' && (route === '/' || route === '/index.html')) {
        return serveStatic(res, path.join(state.publicDir, 'index.html'))
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

// CLI entry: only runs when invoked directly via `npx tsx server.ts`.
// Tests import startServer/stopServer and do their own setup; the block
// below only fires when this file is the process entry point.
const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(__filename)
if (isMain) {
  const PORT = Number(process.env.PORT ?? 7878)
  const EXAMPLE_DIR = __dirname
  startServer({
    port:       PORT,
    exampleDir: EXAMPLE_DIR,
    agentFile:  path.join(EXAMPLE_DIR, 'agents', 'sanguo-researcher.md'),
    corpusRoot: path.join(EXAMPLE_DIR, 'corpus'),
  }).then(() => {
    console.log(`agent-docs-qa playground at http://localhost:${PORT}`)
  })
}
