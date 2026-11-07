import { randomUUID } from 'node:crypto'
import type { Readable, Writable } from 'node:stream'
import type { ExecutionClient } from './ExecutionClient.js'
import { ExecutionError, type ExecutionRecord, type ToolCall, type ToolResult } from './types.js'

export const EXECUTION_PROTOCOL_VERSION = 1
const MAX_FRAME = 8 * 1024 * 1024
const MAX_OUTPUT = 16 * 1024 * 1024
const MAX_PENDING = 64

type Id = string | number
type Params = Record<string, unknown>
type Client = Pick<ExecutionClient, 'capabilities' | 'createContext' | 'getContext' | 'start' | 'query' | 'cancel' | 'toolCall' | 'pendingToolCalls' | 'reconcile' | 'observations' | 'purgeObservations'>
class RpcError extends Error {
  constructor(readonly code: number, message: string) { super(message) }
}
function object(value: unknown): value is Params {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}
function validId(id: unknown): id is Id {
  return (typeof id === 'string' && id.length > 0 && id.length <= 128) || (typeof id === 'number' && Number.isSafeInteger(id))
}
function params(value: unknown, allowed: string[], required: string[] = []): Params {
  const p = value === undefined ? {} : value
  if (!object(p) || Object.keys(p).some(k => !allowed.includes(k)) || required.some(k => !(k in p))) throw new RpcError(-32602, 'Invalid params.')
  return p
}
function string(p: Params, key: string): string {
  if (typeof p[key] !== 'string' || !p[key]) throw new RpcError(-32602, 'Invalid params.')
  return p[key]
}

/** JSON-RPC over bounded JSONL. The client must use lifetime: 'host'. */
export function serveExecutionSidecar(input: Readable, output: Writable, client: Client, signal?: AbortSignal): Promise<void> {
  return new Promise(resolve => {
    let initialized = false, closing = false, buffer = ''
    const inFlight = new Set<Id>()
    const owned = new Set<string>()
    const watched = new Set<string>()
    const observationWatch = new Map<string, { cursor: number; status?: string }>()
    const tools = new Map<Id, (result: ToolResult) => void>()
    const close = () => {
      if (closing) return
      closing = true
      clearInterval(pulse)
      signal?.removeEventListener('abort', close)
      input.off('data', onData)
      // Cancellation stays concurrent with pending tool calls. Never replay a tool.
      void Promise.allSettled([...owned].map(runId => Promise.resolve().then(() => client.cancel(runId)))).then(() => {
        for (const done of tools.values()) done({ ok: false, code: 'rejected', message: 'Host disconnected.' })
        tools.clear()
        if (output.destroyed || output.writableFinished) { resolve(); return }
        // Flush already accepted replies on normal EOF; never hang shutdown if
        // the host stopped draining stdout. A broken channel remains unknown.
        const timer = setTimeout(resolve, 1000)
        try { output.write('', () => { clearTimeout(timer); resolve() }) }
        catch { clearTimeout(timer); resolve() }
      })
    }
    const send = (message: unknown): boolean => {
      if (closing) return false
      const line = JSON.stringify(message) + '\n'
      if (Buffer.byteLength(line) > MAX_FRAME || output.writableLength + Buffer.byteLength(line) > MAX_OUTPUT) { close(); return false }
      try { output.write(line); return true } catch { close(); return false }
    }
    const error = (id: Id | null, code: number, message: string, data?: unknown) => {
      send({ jsonrpc: '2.0', id, error: { code, message, ...(data ? { data } : {}) } })
    }
    const onTool = (call: ToolCall): Promise<ToolResult> => new Promise(done => {
      if (closing || tools.size >= MAX_PENDING) { close(); done({ ok: false, code: 'rejected', message: 'Host unavailable.' }); return }
      const id = `tool:${randomUUID()}`
      tools.set(id, done)
      send({ jsonrpc: '2.0', id, method: 'tool.call', params: call })
    })
    const dispatch = async (method: string, raw: unknown): Promise<unknown> => {
      if (method === 'initialize') {
        const p = params(raw, ['protocolVersion'], ['protocolVersion'])
        if (initialized) throw new RpcError(-32600, 'Already initialized.')
        if (p.protocolVersion !== EXECUTION_PROTOCOL_VERSION) throw new RpcError(-32001, 'Incompatible protocol version.')
        initialized = true
        return { protocolVersion: EXECUTION_PROTOCOL_VERSION, capabilities: client.capabilities() }
      }
      if (!initialized) throw new RpcError(-32002, 'Initialize first.')
      switch (method) {
        case 'capabilities': params(raw, []); return client.capabilities()
        case 'context.create': {
          const p = params(raw, ['cwd', 'storage'], ['cwd'])
          if (p.storage !== undefined) {
            const s = params(p.storage, ['configDir', 'sessionDir'], ['configDir', 'sessionDir'])
            string(s, 'configDir'); string(s, 'sessionDir')
          }
          return client.createContext(string(p, 'cwd'), p.storage as Parameters<Client['createContext']>[1])
        }
        case 'context.get': { const p = params(raw, ['contextId'], ['contextId']); return client.getContext(string(p, 'contextId')) }
        case 'execution.start': {
          const p = params(raw, ['contextId', 'input', 'constraints'], ['contextId', 'input'])
          const constraints = p.constraints as Parameters<Client['start']>[2]
          if (watched.size >= MAX_PENDING) throw new RpcError(-32600, 'Too many active executions.')
          const runId = client.start(string(p, 'contextId'), string(p, 'input'), constraints, constraints?.tools !== undefined ? onTool : undefined)
          owned.add(runId); watched.add(runId)
          if (constraints?.observation) observationWatch.set(runId, { cursor: 0 })
          return { runId }
        }
        case 'execution.query': { const p = params(raw, ['runId'], ['runId']); return client.query(string(p, 'runId')) ?? null }
        case 'execution.cancel': { const p = params(raw, ['runId'], ['runId']); return client.cancel(string(p, 'runId')) }
        case 'observation.read': {
          const p = params(raw, ['runId', 'afterSequence', 'limit'], ['runId'])
          return client.observations(string(p, 'runId'), p.afterSequence as number | undefined, p.limit as number | undefined)
        }
        case 'observation.purge': { const p = params(raw, ['runId'], ['runId']); return client.purgeObservations(string(p, 'runId')) }
        case 'tool.get': { const p = params(raw, ['callId'], ['callId']); return client.toolCall(string(p, 'callId')) ?? null }
        case 'tool.pending': { const p = params(raw, ['contextId'], ['contextId']); return client.pendingToolCalls(string(p, 'contextId')) }
        case 'tool.reconcile': { const p = params(raw, ['callId', 'output'], ['callId', 'output']); return client.reconcile(string(p, 'callId'), string(p, 'output')) }
        default: throw new RpcError(-32601, 'Method not found.')
      }
    }
    const line = (text: string) => {
      if (closing) return
      let m: unknown
      try { m = JSON.parse(text) } catch { error(null, -32700, 'Parse error.'); return }
      if (!object(m) || m.jsonrpc !== '2.0' || !validId(m.id)) { error(null, -32600, 'Invalid request.'); return }
      const id = m.id
      if (!('method' in m)) {
        const done = tools.get(id)
        // Unknown/late responses cannot be interpreted as new commands.
        if (!done) return
        if (Object.keys(m).some(k => !['jsonrpc', 'id', 'result', 'error'].includes(k)) || ('result' in m) === ('error' in m)) { error(id, -32600, 'Invalid response.'); return }
        tools.delete(id)
        done('error' in m ? { ok: false, code: 'rejected', message: 'Host rejected the tool call.' } : m.result as ToolResult)
        return
      }
      if (typeof m.method !== 'string' || Object.keys(m).some(k => !['jsonrpc', 'id', 'method', 'params'].includes(k))) { error(id, -32600, 'Invalid request.'); return }
      if (inFlight.has(id)) { error(id, -32600, 'Duplicate request id.'); close(); return }
      if (inFlight.size >= MAX_PENDING) { error(id, -32600, 'Too many requests.'); close(); return }
      inFlight.add(id)
      // dispatch runs synchronously through start/initialize before it can yield.
      void dispatch(m.method, m.params).then(result => { send({ jsonrpc: '2.0', id, result }) }, e => {
        if (e instanceof RpcError) error(id, e.code, e.message)
        else if (e instanceof ExecutionError) error(id, -32000, 'Execution request failed.', { code: e.code })
        else error(id, -32603, 'Internal error.')
      }).finally(() => { inFlight.delete(id) })
    }
    const onData = (chunk: string) => {
      buffer += chunk
      let end: number
      while (!closing && (end = buffer.indexOf('\n')) >= 0) {
        const frame = buffer.slice(0, end); buffer = buffer.slice(end + 1)
        if (Buffer.byteLength(frame) > MAX_FRAME) { error(null, -32600, 'Frame too large.'); close(); return }
        line(frame)
      }
      if (Buffer.byteLength(buffer) > MAX_FRAME) { error(null, -32600, 'Frame too large.'); close() }
    }
    const pulse = setInterval(() => {
      if (closing) return
      try {
        for (const [runId, cursor] of observationWatch) {
          const page = client.observations(runId, cursor.cursor, 100)
          for (const event of page.events) send({ jsonrpc: '2.0', method: 'execution.observation', params: event })
          cursor.cursor = page.nextSequence
          const status = JSON.stringify(page.status)
          if (status !== cursor.status) {
            send({ jsonrpc: '2.0', method: 'execution.observation.status', params: { runId, ...page.status } })
            cursor.status = status
          }
          if (client.query(runId)?.finishedAt && page.events.length < 100) observationWatch.delete(runId)
        }
        for (const runId of watched) {
          const r: ExecutionRecord | undefined = client.query(runId)
          if (!r) continue
          const { contextId, status, heartbeatAt, stopped, finishedAt, code } = r
          send({ jsonrpc: '2.0', method: 'execution.updated', params: { runId, contextId, status, heartbeatAt, stopped, finishedAt, code } })
          // A stale heartbeat can report unknown before a genuine terminal record exists.
          if (r.finishedAt !== undefined) { watched.delete(runId); owned.delete(runId) }
        }
      } catch { close() }
    }, 250)
    signal?.addEventListener('abort', close, { once: true })
    if (signal?.aborted) { close(); return }
    input.setEncoding('utf8')
    input.on('data', onData)
    input.once('end', () => { if (buffer.trim()) error(null, -32600, 'Unterminated frame.'); close() })
    input.once('error', close)
    output.once('error', close)
    output.once('close', close)
  })
}
