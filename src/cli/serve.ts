import http, { type IncomingMessage, type ServerResponse, type Server } from 'http'
import { Milkie } from '../runtime/Milkie.js'
import { BroadcastingEventStore } from '../trace/BroadcastingEventStore.js'
import { MemoryStore } from '../store/MemoryStore.js'
import { MemoryEventStore } from '../trace/MemoryEventStore.js'
import { SQLiteStore } from '../store/SQLiteStore.js'
import { JsonlEventStore } from '../trace/JsonlEventStore.js'
import path from 'path'
import type { IStateStore } from '../types/store.js'
import type { IEventStore } from '../trace/EventStore.js'
import type { AgentResult, JSONValue, Message } from '../types/common.js'
import type { ModelEvent, ModelResponse } from '../types/model.js'
import type { PortableSession } from '../runtime/PortableSession.js'

/**
 * #86: `milkie serve` HTTP + SSE sidecar. Built (not started) by this factory so
 * tests can listen on an ephemeral port without spawning a process; the CLI
 * wrapper (see `serveAction`) listens, prints the readiness marker, and binds
 * its lifecycle to the parent via SIGTERM / stdin close.
 *
 * Endpoints (see docs/design/86-milkie-serve.md):
 *   GET  /health                              → { ok: true }
 *   POST /chat   { contextId, goal?, input }  → text/event-stream (D2)
 *   POST /interrupt { contextId }             → { signaled: true }
 *   POST /resume { contextId, input? }        → text/event-stream
 *   POST /llm { system?, messages, stream? }  → JSON (or SSE when stream) (#124)
 *   POST /session/export { contextId }        → PortableSession JSON (#124)
 *   POST /session/import { session }          → { contextId } (#124)
 *   POST /session/history { contextId }       → { messages: Message[] } full transcript (#128)
 */
export interface ServeOptions {
  milkie:      Milkie
  agentId:     string
  /** The Milkie's eventStore, wrapped so persistent events fan out by contextId. */
  broadcaster: BroadcastingEventStore
}

/**
 * Persistent trace events forwarded onto the SSE stream. `agent.run.completed`
 * is deliberately excluded: the terminal event is synthesized from invoke's
 * AgentResult (§7.2) so the stream's closing frame is deterministic and carries
 * the final output, independent of broadcast timing.
 */
const STREAM_EVENT_WHITELIST: ReadonlySet<string> = new Set([
  'agent.run.started',
  'tool.requested', 'tool.responded',
  'skill.loaded', 'skill.unloaded',
  'fsm.transition',
  'agent.spawned', 'agent.returned',
])

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = ''
    req.on('data', c => { data += c })
    req.on('end', () => resolve(data))
    req.on('error', reject)
  })
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { 'content-type': 'application/json' })
  res.end(JSON.stringify(body))
}

function writeSSE(res: ServerResponse, event: string, data: unknown): void {
  if (res.writableEnded || res.destroyed) return
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
}

/** Concatenate the text content of a model response (#124 /llm output). */
function outputText(result: ModelResponse): string {
  return result.content.filter((c): c is { type: 'text'; text: string } => c.type === 'text').map(c => c.text).join('')
}

export function createServeServer(opts: ServeOptions): Server {
  const { milkie, agentId, broadcaster } = opts

  /**
   * Drive one turn (invoke or resume) and stream it as SSE. The stream forwards
   * whitelisted persistent events + token-level message_delta, then closes with
   * a synthesized `agent.run.completed` terminal frame. Errors surface as an
   * `error` frame + an `error` terminal — never a bare disconnect (§7).
   */
  async function streamTurn(
    res: ServerResponse,
    contextId: string,
    run: (onModelEvent: (e: ModelEvent) => void) => Promise<AgentResult>,
  ): Promise<void> {
    res.writeHead(200, {
      'content-type':  'text/event-stream',
      'cache-control': 'no-cache',
      'connection':    'keep-alive',
    })
    const unsub = broadcaster.subscribe(contextId, ev => {
      if (STREAM_EVENT_WHITELIST.has(ev.type)) writeSSE(res, ev.type, ev.payload)
    })
    // If the client disconnects mid-run, release the subscription immediately
    // (the run keeps going — a later /resume can continue it) and swallow any
    // late write errors on the dead socket so they can't crash the process.
    res.on('close', unsub)
    res.on('error', () => { /* client-side disconnect; writes are guarded by writeSSE */ })
    try {
      const result = await run(e => {
        if (e.type === 'message_delta') writeSSE(res, 'message_delta', e.data)
      })
      writeSSE(res, 'agent.run.completed', { status: result.status, output: result.output })
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      writeSSE(res, 'error', { message })
      writeSSE(res, 'agent.run.completed', { status: 'error', output: '', error: message })
    } finally {
      unsub()
      if (!res.writableEnded && !res.destroyed) res.end()
    }
  }

  async function handleChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const { contextId, goal, input } = JSON.parse(await readBody(req)) as { contextId?: string; goal?: string; input?: string }
    if (!contextId) return sendJson(res, 400, { error: 'contextId is required' })
    await streamTurn(res, contextId, onModelEvent =>
      milkie.invoke({ agentId, goal: goal ?? input ?? '', input: input ?? '', contextId, onModelEvent }),
    )
  }

  async function handleInterrupt(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const { contextId } = JSON.parse(await readBody(req)) as { contextId?: string }
    if (!contextId) return sendJson(res, 400, { error: 'contextId is required' })
    await milkie.interrupt(contextId)
    sendJson(res, 200, { contextId, signaled: true })
  }

  async function handleResume(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const { contextId, input } = JSON.parse(await readBody(req)) as { contextId?: string; input?: string }
    if (!contextId) return sendJson(res, 400, { error: 'contextId is required' })
    await streamTurn(res, contextId, onModelEvent =>
      milkie.resume(`context:${contextId}:checkpoint:latest`, agentId, 'continue', input ?? 'continue', { onModelEvent }),
    )
  }

  // #83: expose session context variables over HTTP so an external (Python)
  // provider can set/get/list them across the process boundary.
  async function handleContextSet(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const { contextId, name, value } = JSON.parse(await readBody(req)) as { contextId?: string; name?: string; value?: JSONValue }
    if (!contextId || !name) return sendJson(res, 400, { error: 'contextId and name are required' })
    await milkie.setContextVar(contextId, name, value as JSONValue)
    sendJson(res, 200, { ok: true })
  }

  async function handleContextGet(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const { contextId, name } = JSON.parse(await readBody(req)) as { contextId?: string; name?: string }
    if (!contextId || !name) return sendJson(res, 400, { error: 'contextId and name are required' })
    const value = await milkie.getContextVar(contextId, name)
    sendJson(res, 200, { value: value ?? null })
  }

  async function handleContextList(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const { contextId } = JSON.parse(await readBody(req)) as { contextId?: string }
    if (!contextId) return sendJson(res, 400, { error: 'contextId is required' })
    const vars = await milkie.listContextVars(contextId)
    sendJson(res, 200, { vars })
  }

  // #124: one-shot LLM completion (alfred's call_llm). Borrows the single loaded
  // agent's model config; bypasses the FSM. stream=true → SSE message_delta… +
  // a `done` terminal (errors as an `error` frame + `done`, never a bare
  // disconnect); otherwise a single JSON { output, usage }.
  async function handleLlm(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const { system, messages, stream, tier, temperature } = JSON.parse(await readBody(req)) as { system?: string; messages?: Message[]; stream?: boolean; tier?: string; temperature?: number }
    if (!Array.isArray(messages) || messages.length === 0) return sendJson(res, 400, { error: 'messages is required' })
    if (!stream) {
      const result = await milkie.complete(agentId, { system, messages, tier, temperature })
      return sendJson(res, 200, { output: outputText(result), usage: result.usage })
    }
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', 'connection': 'keep-alive' })
    res.on('error', () => { /* client disconnect; writes are guarded by writeSSE */ })
    try {
      const result = await milkie.complete(agentId, { system, messages, tier, temperature }, e => {
        if (e.type === 'message_delta') writeSSE(res, 'message_delta', e.data)
      })
      writeSSE(res, 'done', { usage: result.usage })
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      writeSSE(res, 'error', { message })
      writeSSE(res, 'done', { error: message })
    } finally {
      if (!res.writableEnded && !res.destroyed) res.end()
    }
  }

  // #124: project the #84 portable-session library API onto HTTP. Library errors
  // map to 4xx (no session → 404; unsupported schemaVersion → 400), else 500.
  async function handleSessionExport(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const { contextId } = JSON.parse(await readBody(req)) as { contextId?: string }
    if (!contextId) return sendJson(res, 400, { error: 'contextId is required' })
    try {
      sendJson(res, 200, await milkie.exportSession(contextId))
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      sendJson(res, /No session to export/.test(message) ? 404 : 500, { error: message })
    }
  }

  // #128: project Milkie.getSessionHistory onto HTTP — the full per-message
  // transcript (every run/turn under the context, tool chains intact). No session
  // → 404, else 500.
  async function handleSessionHistory(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const { contextId } = JSON.parse(await readBody(req)) as { contextId?: string }
    if (!contextId) return sendJson(res, 400, { error: 'contextId is required' })
    try {
      sendJson(res, 200, { messages: await milkie.getSessionHistory(contextId) })
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      sendJson(res, /No session/.test(message) ? 404 : 500, { error: message })
    }
  }

  async function handleSessionImport(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const { session } = JSON.parse(await readBody(req)) as { session?: PortableSession }
    if (!session) return sendJson(res, 400, { error: 'session is required' })
    try {
      sendJson(res, 200, await milkie.importSession(session))
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      sendJson(res, /schemaVersion/.test(message) ? 400 : 500, { error: message })
    }
  }

  return http.createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? '/', 'http://localhost')
        const route = url.pathname
        if (req.method === 'GET'  && route === '/health')    return sendJson(res, 200, { ok: true })
        if (req.method === 'POST' && route === '/chat')      return await handleChat(req, res)
        if (req.method === 'POST' && route === '/interrupt') return await handleInterrupt(req, res)
        if (req.method === 'POST' && route === '/resume')    return await handleResume(req, res)
        if (req.method === 'POST' && route === '/context/set')  return await handleContextSet(req, res)
        if (req.method === 'POST' && route === '/context/get')  return await handleContextGet(req, res)
        if (req.method === 'POST' && route === '/context/list') return await handleContextList(req, res)
        if (req.method === 'POST' && route === '/llm')            return await handleLlm(req, res)
        if (req.method === 'POST' && route === '/session/history') return await handleSessionHistory(req, res)
        if (req.method === 'POST' && route === '/session/export') return await handleSessionExport(req, res)
        if (req.method === 'POST' && route === '/session/import') return await handleSessionImport(req, res)
        sendJson(res, 404, { error: `no route ${req.method} ${route}` })
      } catch (err) {
        sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) })
      }
    })()
  })
}

/**
 * Listen, print the readiness marker, and resolve only when the server is
 * gracefully shut down (SIGTERM / SIGINT / stdin close) — binding the process
 * lifecycle to its parent (D4). The readiness line is written directly to
 * process.stdout (not buffered through the CLI's MainResult) so the parent can
 * discover the port immediately even though this command never "returns" until
 * shutdown.
 */
export function runServeServer(server: Server, opts: { port: number; host?: string }): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const onError = (err: Error): void => reject(err)
    server.on('error', onError)
    server.listen(opts.port, opts.host ?? '127.0.0.1', () => {
      const addr = server.address() as { port: number }
      process.stdout.write(`MILKIE_SERVE_READY ${addr.port}\n`)
    })
    let closing = false
    const shutdown = (): void => {
      if (closing) return
      closing = true
      // Detach every listener we registered so repeated serve sessions in one
      // process don't accumulate handlers, and release the event loop —
      // stdin.resume() (below) keeps the process alive to watch for stdin close,
      // so it must be paused for the process to exit once the server closes.
      process.off('SIGTERM', shutdown)
      process.off('SIGINT', shutdown)
      process.stdin.off('end', shutdown)
      process.stdin.off('close', shutdown)
      server.off('error', onError)
      process.stdin.pause()
      server.close(() => resolve())
      // Force-resolve if connections linger past a grace period.
      setTimeout(() => resolve(), 1000).unref()
    }
    process.on('SIGTERM', shutdown)
    process.on('SIGINT', shutdown)
    process.stdin.on('end', shutdown)
    process.stdin.on('close', shutdown)
    process.stdin.resume()
  })
}

/**
 * #130: pick serve's persistence backend. Default is in-memory (sessions live in
 * this process only — the #86 behavior). `state-store=sqlite` uses a persistent
 * SQLite stateStore + Jsonl eventStore under `data-dir`, so a restarted sidecar
 * recovers a context from its event-sourced checkpoint (#73). The two stores are
 * coupled on purpose: recovery needs both the checkpoint events (eventStore) and
 * the `checkpoint-run:latest` pointer (stateStore), so they persist together.
 */
export async function buildServeStores(opts: { stateStore?: 'memory' | 'sqlite'; dataDir?: string }): Promise<{ stateStore: IStateStore; eventStore: IEventStore }> {
  const kind = opts.stateStore ?? 'memory'
  if (kind === 'memory') {
    return { stateStore: new MemoryStore(), eventStore: new MemoryEventStore() }
  }
  if (kind === 'sqlite') {
    if (!opts.dataDir) throw new Error('--data-dir is required when --state-store=sqlite')
    const stateStore = new SQLiteStore({ path: path.join(opts.dataDir, 'state.sqlite') })
    await stateStore.init()
    const eventStore = new JsonlEventStore(path.join(opts.dataDir, 'runs'))
    return { stateStore, eventStore }
  }
  // Fail fast: an unknown backend (typo, or unsupported like redis) must NOT
  // silently fall back to memory — the caller would think persistence is on and
  // lose sessions on restart.
  throw new Error(`unknown --state-store "${String(kind)}"; expected "memory" or "sqlite"`)
}

/**
 * CLI entry for `milkie serve`: load the agent file, build a process-local
 * Milkie (in-memory by default, or persistent SQLite+Jsonl under `--data-dir` so
 * the same contextId recovers from checkpoint after a restart — D5/#130), and
 * serve until shut down. The gateway is resolved from the agent's own model config.
 */
export async function serveMain(opts: { agent: string; port: number; host?: string; stateStore?: 'memory' | 'sqlite'; dataDir?: string }): Promise<void> {
  const { stateStore, eventStore } = await buildServeStores(opts)
  const broadcaster = new BroadcastingEventStore(eventStore)
  const milkie = new Milkie({ stateStore, eventStore: broadcaster })
  const config = milkie.loadAgentFile(opts.agent)
  const server = createServeServer({ milkie, agentId: config.agentId, broadcaster })
  await runServeServer(server, { port: opts.port, host: opts.host })
}
