import { createConnection, type Socket } from 'node:net'
import { readFileSync } from 'node:fs'

const socketPath = process.argv[2]
const toolsFile = process.argv[3]
if (!socketPath || !toolsFile) {
  process.stderr.write('Host tool server requires a socket path and a tools file.\n')
  process.exit(1)
}
const bridgePath: string = socketPath
const tools = JSON.parse(readFileSync(toolsFile, 'utf8')) as Array<{ name: string; description: string; inputSchema: unknown }>
let buffer = Buffer.alloc(0)
let writing: Promise<void> = Promise.resolve()
let bridge: Socket | undefined
let bridgeBuffer = ''
const pending = new Map<string, (message: { ok?: boolean; output?: string; code?: string; message?: string }) => void>()

function send(value: unknown): void {
  // Grok's MCP client parses one JSON value per line and rejects Content-Length headers.
  const packet = `${JSON.stringify(value)}\n`
  writing = writing.then(() => new Promise(resolve => { process.stdout.write(packet, () => resolve()) }))
}
function bridgeCall(id: string, name: string, input: unknown): Promise<{ ok?: boolean; output?: string; code?: string; message?: string }> {
  const client = bridge ?? createConnection(bridgePath)
  if (!bridge) {
    bridge = client
    client.setEncoding('utf8')
    client.on('data', chunk => {
      bridgeBuffer += chunk
      let newline: number
      while ((newline = bridgeBuffer.indexOf('\n')) >= 0) {
        const line = bridgeBuffer.slice(0, newline)
        bridgeBuffer = bridgeBuffer.slice(newline + 1)
        if (!line) continue
        const message = JSON.parse(line) as { id?: string; ok?: boolean; output?: string; code?: string; message?: string }
        const waiter = message.id ? pending.get(message.id) : undefined
        if (waiter && message.id) { pending.delete(message.id); waiter(message) }
      }
    })
    client.on('error', () => { for (const waiter of pending.values()) waiter({ ok: false, code: 'rejected', message: 'Tool bridge failed.' }); pending.clear() })
  }
  return new Promise((resolve, reject) => {
    pending.set(id, resolve)
    client.write(`${JSON.stringify({ id, name, input })}\n`, error => { if (error) reject(error) })
  })
}
async function callTool(message: { id?: unknown; params?: { name?: unknown; arguments?: unknown } }): Promise<void> {
  const name = message.params?.name
  if (typeof name !== 'string') {
    send({ jsonrpc: '2.0', id: message.id, result: { content: [{ type: 'text', text: 'rejected: Tool is not registered.' }], isError: true } })
    return
  }
  let input: unknown = message.params?.arguments ?? {}
  if (typeof input === 'string') input = JSON.parse(input) as unknown
  // The host bridge owns size validation and the persistent rejection record.
  // Frames within its bounded slack must reach it even when over the limit.
  const result = await bridgeCall(String(message.id), name, input)
  const text = result.ok ? result.output : `${result.code}: ${result.message}`
  send({ jsonrpc: '2.0', id: message.id, result: { content: [{ type: 'text', text }], isError: result.ok !== true } })
}
function handle(message: { id?: unknown; method?: string; params?: { name?: unknown; arguments?: unknown } }): void {
  if (!message.method || message.method.startsWith('notifications/')) return
  if (message.method === 'initialize') {
    send({ jsonrpc: '2.0', id: message.id, result: { protocolVersion: '2024-11-05', capabilities: { tools: {} }, serverInfo: { name: 'milkie', version: '1' } } })
    return
  }
  if (message.method === 'ping') { send({ jsonrpc: '2.0', id: message.id, result: {} }); return }
  if (message.method === 'tools/list') {
    send({ jsonrpc: '2.0', id: message.id, result: { tools: tools.map(tool => ({ name: tool.name, description: tool.description, inputSchema: tool.inputSchema })) } })
    return
  }
  if (message.method === 'tools/call') { void callTool(message).catch(() => send({ jsonrpc: '2.0', id: message.id, result: { content: [{ type: 'text', text: 'rejected: Tool bridge failed.' }], isError: true } })); return }
  if (message.id !== undefined) send({ jsonrpc: '2.0', id: message.id, error: { code: -32601, message: 'Method not found.' } })
}
function take(): void {
  for (;;) {
    if (buffer.length === 0) return
    if (buffer[0] === 0x7b) {
      const newline = buffer.indexOf(0x0a)
      if (newline < 0) {
        if (buffer.length > 2 * 1024 * 1024 + 64 * 1024) process.exit(1)
        return
      }
      if (newline > 2 * 1024 * 1024 + 64 * 1024) process.exit(1)
      const line = buffer.subarray(0, newline).toString('utf8')
      buffer = buffer.subarray(newline + 1)
      handle(JSON.parse(line) as { id?: unknown; method?: string })
      continue
    }
    const headerEnd = buffer.indexOf('\r\n\r\n')
    if (headerEnd < 0) return
    const length = Number(/Content-Length:\s*(\d+)/i.exec(buffer.subarray(0, headerEnd).toString('utf8'))?.[1])
    if (!Number.isInteger(length) || length < 0 || length > 2 * 1024 * 1024 + 64 * 1024) process.exit(1)
    const start = headerEnd + 4
    if (buffer.length < start + length) return
    const body = buffer.subarray(start, start + length).toString('utf8')
    buffer = buffer.subarray(start + length)
    handle(JSON.parse(body) as { id?: unknown; method?: string })
  }
}
process.stdin.on('data', (chunk: Buffer) => { buffer = Buffer.concat([buffer, chunk]); try { take() } catch { process.exit(1) } })
process.stdin.on('end', () => process.exit(0))
