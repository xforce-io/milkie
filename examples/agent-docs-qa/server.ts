import http, { type IncomingMessage, type ServerResponse, type Server } from 'http'
import { promises as fs } from 'fs'
import { existsSync, mkdirSync } from 'fs'
import path from 'path'
import { v4 as uuidv4 } from 'uuid'
import { Milkie } from '../../src/runtime/Milkie.js'
import { MemoryStore } from '../../src/store/MemoryStore.js'
import { JsonlEventStore } from '../../src/trace/JsonlEventStore.js'
import { FileTraceObjectStore } from '../../src/trace/TraceObjectStore.js'
import { regionReuseCounts } from '../../src/trace/RegionContextView.js'
import { renderViewer } from '../../src/trace/render/viewer.js'
import { buildExecutionProjection } from '../../src/trace/diagnostics/buildExecutionProjection.js'
import { ReplayError } from '../../src/trace/ReplayError.js'
import { ReplayDivergenceError } from '../../src/trace/ReplayDivergenceError.js'
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
  milkie:           Milkie
  eventStore:       BroadcastingEventStore
  traceObjectStore: FileTraceObjectStore
  runsDir:          string
  publicDir:        string
  corpusRoot:       string
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

async function handleSseStream(
  req: IncomingMessage, res: ServerResponse, s: ServerState, contextId: string,
): Promise<void> {
  res.writeHead(200, {
    'content-type':  'text/event-stream',
    'cache-control': 'no-cache',
    'connection':    'keep-alive',
    'x-accel-buffering': 'no',
  })

  // 1. Catch-up: send all past events for this context
  const past = await readEventsForContext(s.runsDir, contextId)
  for (const e of past) {
    res.write(`data: ${JSON.stringify(e)}\n\n`)
  }

  // If no past events AND no current active run for this contextId,
  // close immediately so the client doesn't hang on an unknown contextId.
  if (past.length === 0) {
    res.end()
    return
  }

  // 2. Subscribe to live events
  const unsubscribe = s.eventStore.subscribe(contextId, (event) => {
    if (!res.writableEnded) res.write(`data: ${JSON.stringify(event)}\n\n`)
  })

  // 3. Cleanup on client disconnect
  req.on('close', () => {
    unsubscribe()
    if (!res.writableEnded) res.end()
  })
}

async function handleReplay(
  res: ServerResponse, s: ServerState, runId: string,
): Promise<void> {
  // Pull the original run's events first — both to confirm the run exists
  // (Milkie.replay would throw ReplayError anyway, but we want a richer
  // response shape with the original answer for client-side comparison)
  // and to compute the run's cumulative cache stats from llm.responded
  // events. Replay itself reads from cache so produces no new usage data;
  // the interesting comparison is "what did the original cost vs what
  // would a replay cost (= zero, all cached)".
  const events = await s.eventStore.readByRunId(runId)
  if (events.length === 0) {
    sendJson(res, 404, { status: 'error', message: 'run not found' })
    return
  }
  const completedEvt = events.find(e => e.type === 'agent.run.completed')
  const originalOutput = (completedEvt?.payload as { lastTextOutput?: string } | undefined)
    ?.lastTextOutput ?? ''

  // Sum cache stats across the original run's LLM responses.
  let origRead = 0
  let origCreated = 0
  let origTotal = 0
  for (const e of events) {
    if (e.type !== 'llm.responded') continue
    const cs = (e.payload as { cacheStats?: { readTokens: number; creationTokens: number; totalInputTokens: number } }).cacheStats
    if (!cs) continue
    origRead    += cs.readTokens
    origCreated += cs.creationTokens
    origTotal   += cs.totalInputTokens
  }

  try {
    const result = await s.milkie.replay(runId)
    sendJson(res, 200, {
      status:          'deterministic',
      replayedOutput:  result.output,
      originalOutput,
      matchesOriginal: result.output === originalOutput,
      originalCacheStats: { readTokens: origRead, creationTokens: origCreated, totalInputTokens: origTotal },
    })
  } catch (err) {
    if (err instanceof ReplayDivergenceError) {
      sendJson(res, 200, {
        status:           'divergent',
        kind:             err.kind,
        summary:          err.summary,
        actualHashPrefix: err.actualHash ? err.actualHash.slice(0, 12) : '',
        availableCount:   err.availableHashes.length,
        originalOutput,
        originalCacheStats: { readTokens: origRead, creationTokens: origCreated, totalInputTokens: origTotal },
      })
    } else if (err instanceof ReplayError) {
      sendJson(res, 400, { status: 'error', message: err.message })
    } else {
      sendJson(res, 500, { status: 'error', message: (err as Error).message })
    }
  }
}

async function handleViewer(
  res: ServerResponse, s: ServerState, runId: string,
): Promise<void> {
  const events = await s.eventStore.readByRunId(runId)
  if (events.length === 0) {
    sendJson(res, 404, { error: 'run not found' })
    return
  }
  // Hydrate region content the same way `milkie trace report` does
  // (cli/main.ts): look up each region's canonical content by hash.
  const regionContent = new Map<string, string>()
  for (const h of regionReuseCounts(events).keys()) {
    const c = await s.traceObjectStore.getCanonical(h)
    if (c !== undefined) regionContent.set(h, c)
  }
  const html = renderViewer(events, { regionContent })
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
  res.end(html)
}

// #70: the Execution tab is a projection over the event log — the core
// buildExecutionProjection owns all attribution (cache tiering, region
// grouping, tool pairing); the frontend only renders this JSON.
async function handleExecution(
  res: ServerResponse, s: ServerState, runId: string,
): Promise<void> {
  const events = await s.eventStore.readByRunId(runId)
  if (events.length === 0) {
    sendJson(res, 404, { error: 'run not found' })
    return
  }
  // Hydrate region content the same way handleViewer / `trace report` does.
  const regionContent = new Map<string, string>()
  for (const h of regionReuseCounts(events).keys()) {
    const c = await s.traceObjectStore.getCanonical(h)
    if (c !== undefined) regionContent.set(h, c)
  }
  sendJson(res, 200, buildExecutionProjection(events, { regionContent }))
}

async function handleSourceFetch(
  res: ServerResponse, s: ServerState, relPath: string, linesQuery: string | null,
): Promise<void> {
  // Resolve under corpusRoot and refuse anything that escapes it.
  // path.resolve normalizes ../, so a "../../etc/passwd" attempt produces
  // an absolute path outside corpusRoot and is rejected by the prefix check.
  const corpusAbs = path.resolve(s.corpusRoot)
  const resolved  = path.resolve(corpusAbs, relPath)
  if (resolved !== corpusAbs && !resolved.startsWith(corpusAbs + path.sep)) {
    sendJson(res, 400, { error: 'path escapes corpus root' })
    return
  }
  let content: string
  try {
    content = await fs.readFile(resolved, 'utf-8')
  } catch {
    sendJson(res, 404, { error: 'source not found' })
    return
  }
  const allLines = content.split(/\r?\n/)
  let startLine = 1
  let endLine   = allLines.length
  if (linesQuery) {
    const m = linesQuery.match(/^(\d+)(?:-(\d+))?$/)
    if (!m) { sendJson(res, 400, { error: 'invalid lines query' }); return }
    startLine = Math.max(1, parseInt(m[1]!, 10))
    endLine   = m[2] ? Math.max(startLine, parseInt(m[2], 10)) : startLine
  }
  const slice = allLines.slice(startLine - 1, endLine).join('\n')
  sendJson(res, 200, { path: relPath, startLine, endLine, content: slice })
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

  const objectsDir = path.join(config.exampleDir, '.milkie', 'objects')
  if (!existsSync(objectsDir)) mkdirSync(objectsDir, { recursive: true })
  const traceObjectStore = new FileTraceObjectStore(objectsDir)

  const eventStore = new BroadcastingEventStore(new JsonlEventStore(runsDir))
  const milkie     = new Milkie({
    stateStore: new MemoryStore(),
    gateway:    config.gateway,   // when omitted, Milkie falls back to createGateway(agent.model) per-invoke
    eventStore,
    traceObjectStore,
    defaultModel: { provider: 'volcengine', model: 'doubao-seed-2-0-pro-260215', adapter: 'openai-compatible' },
  })

  for (const tool of makeCorpusToolDefinitions(config.corpusRoot)) {
    milkie.registerTool(tool)
  }
  milkie.loadAgentFile(config.agentFile)
  milkie.loadStandardAgents()   // built-in diagnoser + read-Trace tools

  // Public dir resolution: prefer co-located public/ when present (production
  // mode using the real example dir), else fall back to this file's
  // sibling public/ (test mode where exampleDir is a tmpDir).
  const colocatedPublic = path.join(config.exampleDir, 'public')
  const fallbackPublic  = path.resolve(__dirname, 'public')
  const publicDir = existsSync(colocatedPublic) ? colocatedPublic : fallbackPublic

  const state: ServerState = { milkie, eventStore, traceObjectStore, runsDir, publicDir, corpusRoot: config.corpusRoot }

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

      const sseMatch = route.match(/^\/conversation\/([^/]+)\/stream$/)
      if (req.method === 'GET' && sseMatch) {
        return handleSseStream(req, res, state, decodeURIComponent(sseMatch[1]!))
      }

      const replayMatch = route.match(/^\/run\/([^/]+)\/replay$/)
      if (req.method === 'POST' && replayMatch) {
        return handleReplay(res, state, decodeURIComponent(replayMatch[1]!))
      }

      const viewerMatch = route.match(/^\/run\/([^/]+)\/viewer$/)
      if (req.method === 'GET' && viewerMatch) {
        return handleViewer(res, state, decodeURIComponent(viewerMatch[1]!))
      }

      const executionMatch = route.match(/^\/run\/([^/]+)\/execution$/)
      if (req.method === 'GET' && executionMatch) {
        return handleExecution(res, state, decodeURIComponent(executionMatch[1]!))
      }

      const sourceMatch = route.match(/^\/source\/(.+)$/)
      if (req.method === 'GET' && sourceMatch) {
        return handleSourceFetch(
          res, state,
          decodeURIComponent(sourceMatch[1]!),
          url.searchParams.get('lines'),
        )
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
