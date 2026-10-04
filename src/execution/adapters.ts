import { openSync, readSync, closeSync, lstatSync, statSync, symlinkSync, realpathSync } from 'node:fs'
import { isAbsolute, join, relative } from 'node:path'
import { GROK_NATIVE_TOOLS, grokLockEnv } from './hostTools.js'
import { ExecutionError, type CliStorage, type ExecutionContext, type ExecutionCode } from './types.js'

const CREDENTIAL_ENV = /(?:^GROK_AUTH$|_API_KEY$|_AUTH_TOKEN$|_OAUTH_TOKEN$|_ACCESS_TOKEN$|_REFRESH_TOKEN$)/
// CLIs read config, plugins, skills and MCP servers from HOME and from their own env overlays,
// so the CLI receives only these host variables and HOME is the dedicated config directory.
const CLI_ENV = /^(?:PATH|LANG|LANGUAGE|LC_[A-Z_]+|TERM|TZ|TMPDIR|TMP|TEMP|USER|LOGNAME|SHELL|(?:HTTP|HTTPS|ALL|NO)_PROXY|(?:http|https|all|no)_proxy|NODE_USE_ENV_PROXY|SSL_CERT_FILE|SSL_CERT_DIR|NODE_EXTRA_CA_CERTS)$/
export interface CliCommand { command: string; args: string[]; stdin?: string }

function directory(path: string | undefined, code: 'config_missing' | 'session_missing'): string {
  if (!path) throw new ExecutionError(code)
  try {
    const real = realpathSync(path)
    if (!statSync(real).isDirectory()) throw new Error('not a directory')
    return real
  } catch (error) {
    if (error instanceof ExecutionError) throw error
    throw new ExecutionError(code)
  }
}
function assertLogin(configDir: string): void {
  const file = join(configDir, 'auth.json')
  let real: string
  try {
    const info = statSync(file)
    if (!info.isFile() || info.size === 0) throw new Error('missing login')
    real = realpathSync(file)
  } catch (error) {
    if (error instanceof ExecutionError) throw error
    throw new ExecutionError('config_missing')
  }
  const fromConfig = relative(configDir, real)
  if (fromConfig.startsWith('..') || isAbsolute(fromConfig)) throw new ExecutionError('config_missing')
}
function bindGrokSessions(configDir: string, sessionDir: string): void {
  const link = join(configDir, 'sessions')
  let info: ReturnType<typeof lstatSync>
  try { info = lstatSync(link) } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw new ExecutionError('storage_error')
    try {
      symlinkSync(sessionDir, link)
      return
    } catch (linkError) {
      // A concurrent context may have bound the same entry first; validate it below.
      if ((linkError as NodeJS.ErrnoException).code !== 'EEXIST') throw new ExecutionError('storage_error')
      try { info = lstatSync(link) } catch { throw new ExecutionError('storage_error') }
    }
  }
  let target: string
  try { target = realpathSync(link) } catch { throw new ExecutionError('invalid_request') }
  if (target !== sessionDir || (!info.isSymbolicLink() && !info.isDirectory())) throw new ExecutionError('invalid_request')
}
/** Resolve host directories, require login material, and point Grok sessions at the session directory. */
export function resolveCliStorage(runtime: string | undefined, storage: CliStorage | undefined): { configDir: string; sessionDir: string } {
  const configDir = directory(storage?.configDir, 'config_missing')
  const sessionDir = directory(storage?.sessionDir, 'session_missing')
  if (configDir === sessionDir) throw new ExecutionError('invalid_request')
  assertLogin(configDir)
  if (runtime === 'grok-cli') bindGrokSessions(configDir, sessionDir)
  return { configDir, sessionDir }
}
export function prepareCliStorage(context: ExecutionContext): void {
  if (context.connection.transport !== 'agent-cli') return
  const storage = resolveCliStorage(context.connection.runtime, { configDir: context.configDir ?? '', sessionDir: context.sessionDir ?? '' })
  if (storage.configDir !== context.configDir || storage.sessionDir !== context.sessionDir) throw new ExecutionError('invalid_request')
}
/** Environment for the supervising worker: the host environment without credentials. */
export function supervisorEnvironment(base: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {}
  for (const [key, value] of Object.entries(base)) {
    if (value !== undefined && !CREDENTIAL_ENV.test(key)) env[key] = value
  }
  return env
}
/** Environment for the CLI itself: allowlisted host variables plus the dedicated storage. */
export function cliEnvironment(base: NodeJS.ProcessEnv, context: ExecutionContext, host?: { socketPath: string; forwarding: 'serial' | 'parallel' }): NodeJS.ProcessEnv {
  if (!context.configDir || !context.sessionDir) throw new ExecutionError('config_missing')
  const env: NodeJS.ProcessEnv = {}
  for (const [key, value] of Object.entries(base)) {
    if (value !== undefined && CLI_ENV.test(key)) env[key] = value
  }
  env.HOME = context.configDir
  if (context.connection.runtime === 'grok-cli') {
    env.GROK_HOME = context.configDir
    env.GROK_LEADER_SOCKET = join(context.configDir, 'leader.sock')
    if (host) Object.assign(env, grokLockEnv())
  } else if (context.connection.runtime === 'pi') {
    env.PI_CODING_AGENT_DIR = context.configDir
    env.PI_CODING_AGENT_SESSION_DIR = context.sessionDir
  } else {
    throw new ExecutionError('unsupported_runtime')
  }
  if (host) {
    env.MILKIE_TOOL_SOCKET = host.socketPath
    env.MILKIE_FORWARDING = host.forwarding
  }
  return env
}
export function nativeFile(context: ExecutionContext): string {
  if (context.connection.runtime === 'pi') return context.nativeSessionFile!
  if (!context.sessionDir || !context.nativeSessionId) throw new ExecutionError('session_missing')
  return join(context.sessionDir, encodeURIComponent(context.cwd), context.nativeSessionId, 'chat_history.jsonl')
}
export function assertNativeSession(context: ExecutionContext): void {
  if (!context.hasExecuted) return
  const file = nativeFile(context)
  try { if (statSync(file).size === 0) throw new Error() } catch { throw new ExecutionError('session_missing') }
  if (context.connection.runtime === 'pi') {
    try {
      const fd = openSync(file, 'r')
      let firstLine: string
      try {
        const bytes = Buffer.alloc(65536)
        const count = readSync(fd, bytes, 0, bytes.length, 0)
        const newline = bytes.subarray(0, count).indexOf(10)
        if (newline < 0) throw new Error()
        firstLine = bytes.subarray(0, newline).toString('utf8')
      } finally { closeSync(fd) }
      const header = JSON.parse(firstLine)
      if (header.type !== 'session' || header.id !== context.nativeSessionId || header.cwd !== context.cwd) throw new Error()
    } catch { throw new ExecutionError('session_missing') }
  }
}
export function cliCommand(context: ExecutionContext, input: string, constraints: { toolPolicy?: 'read-only' | 'standard'; timeoutMs: number; maxModelIterations?: number }, promptFile: string, host?: { names?: string[]; extensionPath?: string }): CliCommand {
  const model = context.connection.model ? ['--model', context.connection.model] : []
  const turnLimit = constraints.maxModelIterations !== undefined ? ['--max-turns', String(constraints.maxModelIterations)] : []
  if (context.connection.runtime === 'pi') {
    if (host?.names && !host.extensionPath) throw new ExecutionError('process_failed')
    const extension = host?.extensionPath ? ['--extension', host.extensionPath] : []
    const tools = host?.names
      ? ['--no-builtin-tools', ...extension, '--tools', host.names.join(',')]
      : ['--tools', constraints.toolPolicy === 'standard' ? 'read,bash,edit,write,grep,find,ls' : 'read,grep,find,ls', ...extension]
    return { command: 'pi', args: ['--print', '--mode', 'json', '--session-dir', context.sessionDir!, '--session', context.nativeSessionFile!, '--no-extensions', '--no-skills', '--no-prompt-templates', '--no-themes', '--no-context-files', '--no-approve', ...tools, ...model], stdin: input }
  }
  if (context.connection.runtime === 'grok-cli') {
    const permissionMode = (host?.names || constraints.toolPolicy === 'standard') ? 'bypassPermissions' : 'plan'
    // Removing every builtin also drops the host MCP tool from the model request. Permission denies block the effect instead.
    const readOnlyDenied = !host?.names && constraints.toolPolicy !== 'standard' ? ['--disallowed-tools', GROK_NATIVE_TOOLS] : []
    // Grok 1.0.41 reaches MCP tools only through search_tool/use_tool. Removing those two also removes the host tool.
    // Out-of-bounds access is closed by the milkie-only server plus these permission denies.
    const hostLock = host?.names ? ['--disallowed-tools', 'x_search,web_search,web_fetch', '--deny', 'Bash(*)', '--deny', 'Read(**)', '--deny', 'Edit(**)', '--deny', 'Grep'] : []
    return { command: 'grok', args: ['--cwd', context.cwd, '--leader-socket', join(context.configDir!, 'leader.sock'), context.hasExecuted ? '--resume' : '--session-id', context.nativeSessionId!, '--output-format', 'streaming-json', '--no-subagents', '--disable-web-search', '--permission-mode', permissionMode, ...readOnlyDenied, ...hostLock, ...turnLimit, ...model, '--prompt-file', promptFile] }
  }
  throw new ExecutionError('unsupported_runtime')
}
export function classifyFailure(text: string): ExecutionCode {
  if (/unauthori[sz]ed|authentication|not logged in|not signed in|login required|no api key|please (?:log|sign) in|invalid.*(?:token|api.key)|\b401\b|expired.*(?:token|credential)/i.test(text)) return 'auth_failed'
  if (/session.*(?:not found|missing|does not exist|invalid)/i.test(text)) return 'session_missing'
  return 'process_failed'
}
/** Parse only the current invocation's final output; never expose raw CLI events. */
export class CliEvents {
  private buffer = ''
  private finalAssistant = false
  output = ''
  sessionId?: string
  ended = false
  code?: ExecutionCode
  constructor(private runtime: 'grok-cli' | 'pi') {}
  push(chunk: string): void {
    this.buffer += chunk
    if (this.buffer.length > 4 * 1024 * 1024) { this.code = 'protocol_error'; this.buffer = ''; return }
    let newline: number
    while ((newline = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, newline); this.buffer = this.buffer.slice(newline + 1)
      this.line(line)
    }
  }
  finish(): void { if (this.buffer.trim()) this.line(this.buffer); this.buffer = '' }
  private line(line: string): void {
    let e: any
    try { e = JSON.parse(line) } catch { if (line.trim()) this.code = 'protocol_error'; return }
    if (!e || typeof e !== 'object') { this.code = 'protocol_error'; return }
    if (this.runtime === 'grok-cli') {
      if (e.type === 'text' && typeof e.data === 'string') this.output += e.data
      if (e.type === 'max_turns_reached' || e.stopReason === 'max_turns' || e.stopReason === 'error_max_turns') {
        this.code = 'iteration_budget_exhausted'
        if (typeof e.sessionId === 'string') this.sessionId = e.sessionId
        this.ended = false
      } else if (e.type === 'end') {
        this.sessionId = typeof e.sessionId === 'string' ? e.sessionId : undefined
        this.ended = e.stopReason === 'end_turn'
        if (!this.ended) this.code = e.stopReason === 'cancelled' ? 'native_cancelled' : 'process_failed'
      }
      if (e.type === 'error') this.code = classifyFailure(JSON.stringify(e))
    } else {
      if (e.type === 'session') this.sessionId = typeof e.id === 'string' ? e.id : undefined
      if (e.type === 'message_end' && e.message?.role === 'assistant' && this.code !== 'protocol_error') {
        const m = e.message
        // Pi may emit an error and then auto-retry. The latest assistant message is the outcome.
        this.finalAssistant = m.stopReason === 'stop'
        this.code = undefined
        if (m.stopReason === 'length') this.code = 'process_failed'
        this.output = Array.isArray(m.content) ? m.content.filter((c: any) => c?.type === 'text' && typeof c.text === 'string').map((c: any) => c.text).join('') : ''
        if (m.stopReason === 'error' || m.stopReason === 'aborted') this.code = classifyFailure(String(m.errorMessage ?? ''))
      }
      if (e.type === 'agent_end') this.ended = this.finalAssistant
    }
    if (this.output.length > 4 * 1024 * 1024) { this.output = ''; this.code = 'protocol_error' }
  }
}
