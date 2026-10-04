import { execFileSync, spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { existsSync, readFileSync, readdirSync, realpathSync, writeFileSync, unlinkSync, chmodSync } from 'node:fs'
import { createServer, Socket } from 'node:net'
import { dirname, join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parse as parseToml } from 'smol-toml'
import { ExecutionError, type HostToolSchema, type HostToolSpec, type ToolCall, type ToolCallRecord, type ToolResult } from './types.js'
import { ExecutionStore } from './store.js'

/** Native tools that must stay unavailable while the host owns the tool surface. */
export const GROK_NATIVE_TOOLS = [
  'run_terminal_command', 'search_replace', 'write', 'spawn_subagent', 'exit_plan_mode', 'enter_plan_mode',
  'kill_command_or_subagent', 'todo_write', 'send_feedback', 'scheduler_create', 'scheduler_delete', 'scheduler_list',
  'get_command_or_subagent_output', 'ask_user_question', 'monitor', 'search_tool', 'use_tool', 'workflow',
  'image_gen', 'image_edit', 'image_to_video', 'reference_to_video',
].join(',')
/** Builtins that stay available in read-only mode. Host-tool runs deny them too. */
export const GROK_READ_TOOLS = ['read_file', 'list_dir', 'grep']
/** Names Grok still exposes when only the model-facing name is denied. */
export const GROK_HOST_TOOL_ALIASES = ['run_terminal_cmd', 'command', 'cmd', 'bash_command', 'task', 'kill_task', 'kill_terminal_command', 'get_task_output', 'get_terminal_command_output', 'send_subagent_message', 'x_search', 'web_search', 'web_fetch', 'code_interpreter']
const RESERVED = new Set([...GROK_NATIVE_TOOLS.split(','), ...GROK_READ_TOOLS, ...GROK_HOST_TOOL_ALIASES, 'read', 'grep', 'find', 'ls', 'bash', 'edit', 'write', 'ask_question'])
/** Raw host-tool input strings and successful results. UTF-8 bytes, not UTF-16 code units. */
export const MAX_TOOL_PAYLOAD_BYTES = 256 * 1024
/** JSON tool request, including escape expansion. Distinct from the raw payload cap. */
export const MAX_ENCODED_TOOL_REQUEST_BYTES = 2 * 1024 * 1024
const MAX_MESSAGE = 1024
const MARKER = '# milkie-host-tools\n'

export function grokLockEnv(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {}
  for (const vendor of ['CURSOR', 'CLAUDE', 'CODEX']) {
    for (const surface of ['SKILLS', 'RULES', 'AGENTS', 'MCPS', 'HOOKS', 'SESSIONS']) env[`GROK_${vendor}_${surface}_ENABLED`] = '0'
  }
  env.GROK_MANAGED_CONFIG = '0'
  env.GROK_MANAGED_MCPS_ENABLED = '0'
  env.GROK_MANAGED_MCP_GATEWAY_TOOLS_ENABLED = '0'
  return env
}
/** CLI-visible catalog. Previously registered names stay listed so a real CLI can deliver them; only `current` is authorized. */
export function visibleHostTools(store: ExecutionStore, contextId: string, current: HostToolSpec[]): HostToolSpec[] {
  const authorized = new Set(current.map(tool => tool.name))
  const revoked = new Map<string, HostToolSpec>()
  const dir = join(store.root, 'runs')
  if (!existsSync(dir)) return [...current]
  for (const entry of readdirSync(dir)) {
    if (!entry.endsWith('.json') || entry.endsWith('.tools.json')) continue
    const run = store.run(entry.slice(0, -'.json'.length))
    if (!run || run.contextId !== contextId) continue
    const toolsPath = join(dir, `${run.runId}.tools.json`)
    if (!existsSync(toolsPath)) continue
    let listed: unknown
    try { listed = JSON.parse(readFileSync(toolsPath, 'utf8')) } catch { throw new ExecutionError('storage_error') }
    if (!Array.isArray(listed)) throw new ExecutionError('storage_error')
    for (const tool of listed) {
      if (!tool || typeof tool !== 'object' || typeof (tool as { name?: unknown }).name !== 'string') throw new ExecutionError('storage_error')
      const name = (tool as { name: string }).name
      if (authorized.has(name) || revoked.has(name)) continue
      revoked.set(name, tool as HostToolSpec)
    }
  }
  const visible = [...current, ...revoked.values()]
  if (visible.length > 32) throw new ExecutionError('invalid_request')
  return visible
}
export function assertHostTools(tools: HostToolSpec[]): void {
  if (tools.length > 32) throw new ExecutionError('invalid_request')
  const names = new Set<string>()
  for (const tool of tools) {
    if (!tool || typeof tool !== 'object') throw new ExecutionError('invalid_request')
    if (typeof tool.name !== 'string' || !/^[a-z][a-z0-9_]{0,40}$/.test(tool.name) || RESERVED.has(tool.name) || names.has(tool.name)) throw new ExecutionError('invalid_request')
    names.add(tool.name)
    if (typeof tool.description !== 'string' || !tool.description.trim() || tool.description.length > 4000) throw new ExecutionError('invalid_request')
    if (tool.inputSchema?.type !== 'object') throw new ExecutionError('invalid_request')
    assertSchema(tool.inputSchema, 0)
  }
}
function assertSchema(schema: HostToolSchema, depth: number): void {
  if (depth > 8 || !schema || typeof schema !== 'object') throw new ExecutionError('invalid_request')
  if (!['object', 'string', 'number', 'integer', 'boolean', 'array'].includes(schema.type)) throw new ExecutionError('invalid_request')
  if (schema.type === 'object') {
    if (schema.properties !== undefined) {
      if (!schema.properties || typeof schema.properties !== 'object' || Array.isArray(schema.properties)) throw new ExecutionError('invalid_request')
      for (const child of Object.values(schema.properties)) assertSchema(child, depth + 1)
    }
    if (schema.required !== undefined && (!Array.isArray(schema.required) || schema.required.some(item => typeof item !== 'string'))) throw new ExecutionError('invalid_request')
    if (schema.additionalProperties !== undefined && typeof schema.additionalProperties !== 'boolean') throw new ExecutionError('invalid_request')
  }
  if (schema.type === 'array') {
    if (!schema.items) throw new ExecutionError('invalid_request')
    assertSchema(schema.items, depth + 1)
  }
}
export function validateToolInput(schema: HostToolSchema, value: unknown): boolean {
  return matchSchema(schema, value)
}
function matchSchema(schema: HostToolSchema, value: unknown): boolean {
  if (schema.type === 'string') return typeof value === 'string'
  if (schema.type === 'number') return typeof value === 'number' && Number.isFinite(value)
  if (schema.type === 'integer') return typeof value === 'number' && Number.isInteger(value)
  if (schema.type === 'boolean') return typeof value === 'boolean'
  if (schema.type === 'array') return Array.isArray(value) && !!schema.items && value.every(item => matchSchema(schema.items!, item))
  if (schema.type !== 'object' || !value || typeof value !== 'object' || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  const properties = schema.properties ?? {}
  for (const key of schema.required ?? []) if (!(key in record)) return false
  for (const key of Object.keys(record)) {
    const child = properties[key]
    if (!child) { if (schema.additionalProperties !== true) return false; continue }
    if (!matchSchema(child, record[key])) return false
  }
  return true
}
export function normalizeToolResult(value: unknown): ToolResult {
  if (!value || typeof value !== 'object') return { ok: false, code: 'rejected', message: 'Handler returned an invalid result.' }
  const result = value as { ok?: unknown; output?: unknown; code?: unknown; message?: unknown }
  if (result.ok === true && typeof result.output === 'string') {
    if (Buffer.byteLength(result.output) <= MAX_TOOL_PAYLOAD_BYTES) return { ok: true, output: result.output }
    return { ok: false, code: 'rejected', message: 'Tool result exceeds 262144 bytes.' }
  }
  if (result.ok === false && (result.code === 'invalid_input' || result.code === 'rejected') && typeof result.message === 'string' && result.message.length > 0 && result.message.length <= MAX_MESSAGE) {
    return { ok: false, code: result.code, message: result.message }
  }
  return { ok: false, code: 'rejected', message: 'Handler returned an invalid result.' }
}
/** Fail closed on anything other than the one milkie MCP server and disabled external imports. */
export function assertGrokInventory(report: unknown, command: string): void {
  if (!report || typeof report !== 'object') throw new ExecutionError('policy_mismatch')
  const body = report as Record<string, unknown>
  const servers = body.mcpServers
  // `target` is the command Grok actually loaded. The reported source path can still point at the host file.
  const server = Array.isArray(servers) ? servers[0] as { name?: string; target?: string } | undefined : undefined
  if (!Array.isArray(servers) || servers.length !== 1 || server?.name !== 'milkie' || server.target !== command) throw new ExecutionError('policy_mismatch')
  for (const key of ['hooks', 'plugins', 'lspServers', 'marketplaces']) {
    const value = body[key]
    if (!Array.isArray(value) || value.length !== 0) throw new ExecutionError('policy_mismatch')
  }
  const skills = body.skills
  // Grok unpacks its own bundled skills into a fresh GROK_HOME during the first run. Project and imported skills still fail.
  if (!Array.isArray(skills) || skills.some(skill => !skill || typeof skill !== 'object' || (skill as { source?: { type?: string } }).source?.type !== 'bundled')) throw new ExecutionError('policy_mismatch')
  const agents = body.agents
  if (!Array.isArray(agents) || agents.some(agent => !agent || typeof agent !== 'object' || (agent as { source?: { type?: string } }).source?.type !== 'builtin')) throw new ExecutionError('policy_mismatch')
  const cells = (body.externalCompat as { cells?: unknown } | undefined)?.cells
  if (!Array.isArray(cells) || cells.length === 0 || cells.some(cell => !cell || typeof cell !== 'object' || (cell as { enabled?: unknown }).enabled !== false)) throw new ExecutionError('policy_mismatch')
  if ((body.permissions as { managedSettingsActive?: unknown } | undefined)?.managedSettingsActive !== false) throw new ExecutionError('policy_mismatch')
}
function tomlString(value: string): string {
  if (/["\\\n]/.test(value)) throw new ExecutionError('storage_error')
  return `"${value}"`
}
/** A project MCP table can replace the host server. Quoted and dotted TOML keys are the same table. */
export function assertProjectGrokConfig(cwd: string): void {
  const file = join(cwd, '.grok', 'config.toml')
  if (!existsSync(file)) return
  let body: unknown
  try { body = parseToml(readFileSync(file, 'utf8')) } catch { throw new ExecutionError('policy_mismatch') }
  if (!body || typeof body !== 'object' || Array.isArray(body) || Object.prototype.hasOwnProperty.call(body, 'mcp_servers')) throw new ExecutionError('policy_mismatch')
}
/** `grok mcp list` reports the command and args actually loaded. Inspect's target omits args. */
export function assertGrokMcpLaunch(report: unknown, command: string, args: string[]): void {
  if (!Array.isArray(report) || report.length !== 1) throw new ExecutionError('policy_mismatch')
  const server = report[0] as { name?: unknown; command?: unknown; args?: unknown; enabled?: unknown }
  if (server?.name !== 'milkie' || server.command !== command || server.enabled === false) throw new ExecutionError('policy_mismatch')
  if (!Array.isArray(server.args) || server.args.length !== args.length || server.args.some((item, index) => item !== args[index])) throw new ExecutionError('policy_mismatch')
}
/** One host-tool run owns the shared Grok config until it exits. A live holder, or a dead one whose stop was not confirmed, fails closed. */
export function acquireGrokConfigLock(configDir: string, runId: string, previousConfirmed: (previousRunId: string) => boolean): () => void {
  const file = join(configDir, 'milkie-host-tools.lock')
  const payload = `${process.pid} ${runId}\n`
  const claim = () => writeFileSync(file, payload, { flag: 'wx', mode: 0o600 })
  try {
    claim()
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw new ExecutionError('storage_error')
    const [pidText, previousRunId] = (existsSync(file) ? readFileSync(file, 'utf8') : '').trim().split(' ')
    const pid = Number(pidText)
    let live = false
    if (Number.isInteger(pid) && pid > 0) {
      try { process.kill(pid, 0); live = true } catch (err) { live = (err as NodeJS.ErrnoException).code === 'EPERM' }
    }
    if (live || !previousRunId || !previousConfirmed(previousRunId)) throw new ExecutionError('policy_mismatch')
    try { unlinkSync(file) } catch { throw new ExecutionError('policy_mismatch') }
    try { claim() } catch { throw new ExecutionError('policy_mismatch') }
  }
  return () => {
    try { if (readFileSync(file, 'utf8') === payload) unlinkSync(file) } catch { /* A raced release leaves the next acquire fail-closed. */ }
  }
}
export function writeGrokHostConfig(configDir: string, script: string, socketPath: string, toolsFile: string): void {
  const file = join(configDir, 'config.toml')
  if (existsSync(file) && !readFileSync(file, 'utf8').startsWith(MARKER)) throw new ExecutionError('policy_mismatch')
  if (!existsSync(script)) throw new ExecutionError('process_failed')
  const body = `${MARKER}[mcp_servers.milkie]\ncommand = ${tomlString(process.execPath)}\nargs = [${[script, socketPath, toolsFile].map(tomlString).join(', ')}]\n`
  writeFileSync(file, body, { mode: 0o600 })
}
/** Pi settings packages install extra tools. Host mode rejects that old config before the model starts. */
export function assertPiHostConfig(configDir: string): void {
  const file = join(configDir, 'settings.json')
  if (!existsSync(file)) return
  let body: unknown
  try { body = JSON.parse(readFileSync(file, 'utf8')) } catch { throw new ExecutionError('policy_mismatch') }
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw new ExecutionError('policy_mismatch')
  const packages = (body as { packages?: unknown }).packages
  if (packages === undefined) return
  if (!Array.isArray(packages) || packages.length !== 0) throw new ExecutionError('policy_mismatch')
}
function readGrok(args: string[], cwd: string, env: NodeJS.ProcessEnv, alive: () => boolean): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const child = spawn('grok', args, { cwd, env, stdio: ['ignore', 'pipe', 'ignore'] })
    let stdout = ''
    const timer = setTimeout(() => { child.kill('SIGKILL'); reject(new ExecutionError('policy_mismatch')) }, 15000)
    const pulse = setInterval(() => { if (!alive()) child.kill('SIGKILL') }, 50)
    const fail = () => { clearTimeout(timer); clearInterval(pulse); reject(new ExecutionError('policy_mismatch')) }
    child.stdout.setEncoding('utf8')
    child.stdout.on('data', (chunk: string) => { stdout = (stdout + chunk).slice(0, 1_000_000) })
    child.on('error', fail)
    child.on('exit', code => {
      clearTimeout(timer); clearInterval(pulse)
      if (!alive() || code !== 0) { reject(new ExecutionError('policy_mismatch')); return }
      try { resolve(JSON.parse(stdout)) } catch { reject(new ExecutionError('policy_mismatch')) }
    })
  })
}
export function inspectGrok(cwd: string, env: NodeJS.ProcessEnv, leaderSocket: string, launch: { command: string; args: string[] }, alive: () => boolean): Promise<void> {
  assertProjectGrokConfig(cwd)
  const grokArgs = ['--leader-socket', leaderSocket]
  return readGrok(['inspect', '--json', ...grokArgs], cwd, env, alive).then(report => {
    assertGrokInventory(report, launch.command)
    return readGrok(['mcp', 'list', '--json', ...grokArgs], cwd, env, alive)
  }).then(report => { assertGrokMcpLaunch(report, launch.command, launch.args) })
}
function resolveTypeboxModule(): string {
  try {
    const bin = execFileSync('/usr/bin/which', ['pi'], { encoding: 'utf8', env: process.env }).trim()
    let dir = dirname(realpathSync(bin))
    for (let i = 0; i < 8; i++) {
      const candidate = join(dir, 'node_modules', 'typebox', 'build', 'index.mjs')
      if (existsSync(candidate)) return pathToFileURL(candidate).href
      const parent = dirname(dir)
      if (parent === dir) break
      dir = parent
    }
  } catch { /* Fixture pi does not load this extension. */ }
  return 'typebox'
}
export function writePiExtension(file: string, tools: HostToolSpec[], socketPath: string | undefined, forwarding: 'serial' | 'parallel', budget?: { limit: number; markerFile: string }): void {
  if (tools.length === 0) {
    if (!budget) throw new ExecutionError('process_failed')
    const source = `import { writeFileSync } from 'node:fs'
const modelIterationBudget = ${budget.limit}
const modelIterationMarker = ${JSON.stringify(budget.markerFile)}
export default function (pi) {
  let modelIterations = 0
  pi.on('before_provider_request', (_event, ctx) => {
    modelIterations += 1
    if (modelIterations > modelIterationBudget) {
      writeFileSync(modelIterationMarker, 'exhausted\\n')
      ctx.abort()
    }
  })
}
`
    writeFileSync(file, source, { mode: 0o600 })
    return
  }
  const spec = JSON.stringify({ socketPath, forwarding, tools, maxModelIterations: budget?.limit ?? 0, markerFile: budget?.markerFile ?? '' })
  const typebox = JSON.stringify(resolveTypeboxModule())
  const fsImport = budget ? `import { writeFileSync } from 'node:fs'\n` : ''
  const iterationHook = budget ? `  let modelIterations = 0
  pi.on('before_provider_request', (_event, ctx) => {
    modelIterations += 1
    if (modelIterations > spec.maxModelIterations) {
      writeFileSync(spec.markerFile, 'exhausted\\n')
      ctx.abort()
    }
  })
` : ''
  const source = `${fsImport}import net from 'node:net'
import { Type } from ${typebox}
const spec = ${spec}
function toType(schema) {
  if (schema.type === 'string') return Type.String()
  if (schema.type === 'number') return Type.Number()
  if (schema.type === 'integer') return Type.Integer()
  if (schema.type === 'boolean') return Type.Boolean()
  if (schema.type === 'array') return Type.Array(toType(schema.items))
  const properties = {}
  for (const [key, value] of Object.entries(schema.properties || {})) {
    const child = toType(value)
    properties[key] = (schema.required || []).includes(key) ? child : Type.Optional(child)
  }
  return Type.Object(properties)
}
let buffer = ''
let socket
const pending = new Map()
function fail(error) {
  for (const waiter of pending.values()) waiter.reject(error)
  pending.clear()
}
function connect() {
  if (socket) return socket
  socket = net.createConnection(spec.socketPath)
  socket.setEncoding('utf8')
  socket.on('data', chunk => {
    buffer += chunk
    let newline
    while ((newline = buffer.indexOf('\\n')) >= 0) {
      const line = buffer.slice(0, newline)
      buffer = buffer.slice(newline + 1)
      if (!line) continue
      const message = JSON.parse(line)
      const waiter = pending.get(message.id)
      if (!waiter) continue
      pending.delete(message.id)
      waiter.resolve(message)
    }
  })
  socket.on('error', fail)
  return socket
}
function roundTrip(payload) {
  const client = connect()
  return new Promise((resolve, reject) => {
    pending.set(payload.id, { resolve, reject })
    // An idle socket must not keep Pi's --print process alive after the turn ends.
    client.ref()
    client.write(JSON.stringify(payload) + '\\n')
  }).finally(() => { if (pending.size === 0) client.unref() })
}
export default function (pi) {
${iterationHook}  for (const tool of spec.tools) {
    pi.registerTool({
      name: tool.name,
      label: tool.name,
      description: tool.description,
      parameters: toType(tool.inputSchema),
      executionMode: spec.forwarding === 'parallel' ? 'parallel' : 'sequential',
      async execute(toolCallId, params) {
        const result = await roundTrip({ id: toolCallId, name: tool.name, nativeCallId: toolCallId, input: params })
        const text = result.ok ? result.output : result.code + ': ' + result.message
        return { content: [{ type: 'text', text }], details: { ok: Boolean(result.ok), code: result.code } }
      },
    })
  }
}
`
  writeFileSync(file, source, { mode: 0o600 })
}
function rawStringBytes(value: unknown): number {
  if (typeof value === 'string') return Buffer.byteLength(value)
  if (Array.isArray(value)) {
    let max = 0
    for (const item of value) max = Math.max(max, rawStringBytes(item))
    return max
  }
  if (value && typeof value === 'object') {
    let max = 0
    for (const item of Object.values(value as Record<string, unknown>)) max = Math.max(max, rawStringBytes(item))
    return max
  }
  return 0
}
/** Host completion is not delivery. A failed write leaves the call pending. */
function deliverReply(socket: Socket, payload: string): Promise<boolean> {
  if (socket.destroyed || !socket.writable) return Promise.resolve(false)
  return new Promise(resolve => {
    let settled = false
    const finish = (ok: boolean) => { if (!settled) { settled = true; resolve(ok) } }
    const onError = () => finish(false)
    socket.once('error', onError)
    try { socket.write(payload, error => { socket.off('error', onError); finish(!error) }) }
    catch { socket.off('error', onError); finish(false) }
  })
}
export interface ToolBridge { socketPath: string; close(): Promise<void> }
export async function openToolBridge(options: {
  tools: HostToolSpec[]
  forwarding: 'serial' | 'parallel'
  runId: string
  contextId: string
  store: ExecutionStore
  alive: () => boolean
  onHost: (call: ToolCall) => Promise<unknown>
  onBroken: (error: unknown) => void
}): Promise<ToolBridge> {
  const socketPath = join('/tmp', `milkie-${options.runId}.sock`)
  try { unlinkSync(socketPath) } catch { /* absent */ }
  let failed = false
  const broken = (error: unknown) => { if (failed) return; failed = true; options.onBroken(error) }
  let tail = Promise.resolve()
  const enqueue = (job: () => Promise<void>) => {
    if (options.forwarding === 'parallel') return job()
    const run = tail.then(job, job)
    tail = run.then(() => undefined, () => undefined)
    return run
  }
  const tools = new Map(options.tools.map(tool => [tool.name, tool]))
  const prepare = (socket: Socket, line: string): (() => Promise<void>) | undefined => {
    const message = JSON.parse(line) as { id?: unknown; name?: unknown; nativeCallId?: unknown; input?: unknown }
    if (typeof message.id !== 'string' || typeof message.name !== 'string') throw new Error('Malformed tool call.')
    const name = message.name
    const nativeCallId = typeof message.nativeCallId === 'string' && message.nativeCallId ? message.nativeCallId : undefined
    const callId = randomUUID()
    const base: ToolCallRecord = { version: 1, callId, name, input: message.input, runId: options.runId, contextId: options.contextId, status: 'pending', ...(nativeCallId ? { nativeCallId } : {}) }
    const tool = tools.get(name)
    const reply = (result: ToolResult) => socket.write(`${JSON.stringify({ id: message.id, ...result })}\n`)
    if (Buffer.byteLength(line) > MAX_ENCODED_TOOL_REQUEST_BYTES || rawStringBytes(message.input) > MAX_TOOL_PAYLOAD_BYTES) {
      const rejected: ToolResult = {
        ok: false,
        code: 'invalid_input',
        message: Buffer.byteLength(line) > MAX_ENCODED_TOOL_REQUEST_BYTES
          ? 'Encoded tool request exceeds 2097152 bytes.'
          : 'Tool input exceeds 262144 bytes.',
      }
      options.store.write('calls', callId, { ...base, status: 'invalid_input', message: rejected.message })
      reply(rejected)
      return
    }
    if (!tool) {
      const rejected: ToolResult = { ok: false, code: 'rejected', message: 'Tool is not registered.' }
      options.store.write('calls', callId, { ...base, status: 'rejected', message: rejected.message })
      reply(rejected)
      return
    }
    if (!validateToolInput(tool.inputSchema, message.input)) {
      const invalid: ToolResult = { ok: false, code: 'invalid_input', message: 'Input does not match the tool schema.' }
      options.store.write('calls', callId, { ...base, status: 'invalid_input', message: invalid.message })
      reply(invalid)
      return
    }
    // Persist before the serial queue so a later frame is recorded even if the host dies during the previous call.
    options.store.write('calls', callId, base)
    return async () => {
      if (!options.alive()) return
      let result: ToolResult
      try { result = normalizeToolResult(await options.onHost({ callId, ...(nativeCallId ? { nativeCallId } : {}), name, input: message.input, runId: options.runId, contextId: options.contextId })) }
      catch (error) { if (!options.alive()) return; throw error }
      if (!options.alive()) return
      const delivered = await deliverReply(socket, `${JSON.stringify({ id: message.id, ...result })}\n`)
      if (!delivered) return
      options.store.write('calls', callId, { ...base, status: result.ok ? 'succeeded' : result.code, ...(result.ok ? { output: result.output } : { message: result.message }) })
    }
  }
  const sockets = new Set<Socket>()
  const server = createServer(socket => {
    sockets.add(socket)
    socket.on('close', () => sockets.delete(socket))
    let buffer = ''
    socket.setEncoding('utf8')
    socket.on('error', () => { /* The CLI closing its end is not a successful tool result. */ })
    socket.on('data', chunk => {
      buffer += chunk
      if (Buffer.byteLength(buffer) > MAX_ENCODED_TOOL_REQUEST_BYTES + 64 * 1024) { socket.destroy(); broken(new Error('Tool call is too large.')); return }
      let newline: number
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline)
        buffer = buffer.slice(newline + 1)
        if (!line) continue
        try {
          const job = prepare(socket, line)
          if (job) void enqueue(job).catch(broken)
        } catch (error) { broken(error) }
      }
    })
  })
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(socketPath, () => { server.removeAllListeners('error'); server.on('error', broken); resolve() })
  })
  chmodSync(socketPath, 0o600)
  return { socketPath, close: () => {
    for (const socket of sockets) socket.destroy()
    server.unref()
    server.close()
    try { unlinkSync(socketPath) } catch { /* absent */ }
    return Promise.resolve()
  } }
}
/** fd 4 is the host-lifetime pipe. EOF means the host process is gone. The returned function must run before process.exit: an open stream on this fd keeps Node from actually exiting. */
export function watchParentPipe(mark: () => void): () => void {
  const socket = new Socket({ fd: 4, readable: true, writable: false })
  socket.on('end', mark)
  socket.on('error', () => { /* An unusable descriptor is not host death. IPC disconnect remains the other signal. */ })
  socket.resume()
  return () => { socket.destroy() }
}
