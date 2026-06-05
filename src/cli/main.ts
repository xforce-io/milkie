import { Command, Option } from 'commander'
import fs from 'fs'
import path from 'path'
import { Milkie } from '../runtime/Milkie.js'
import { JsonlEventStore } from '../trace/JsonlEventStore.js'
import { FileTraceObjectStore } from '../trace/TraceObjectStore.js'
import { regionReuseCounts } from '../trace/RegionContextView.js'
import { SQLiteStore } from '../store/SQLiteStore.js'
import { MemoryStore } from '../store/MemoryStore.js'
import { checkpointFromEvents } from '../trace/diagnostics/checkpointFromEvents.js'
import { serveMain } from './serve.js'

function findMilkieDir(startDir: string): string | undefined {
  let dir = startDir
  while (true) {
    const candidate = path.join(dir, '.milkie')
    if (fs.existsSync(candidate)) return candidate
    const parent = path.dirname(dir)
    if (parent === dir) return undefined
    dir = parent
  }
}

/**
 * #144: resolve the trace root holding `runs/` (and `objects/`). An explicit
 * `--data-dir` — the same directory passed to `serve --data-dir`, where the
 * sidecar persisted `<dir>/runs/<runId>.jsonl` — takes precedence and bypasses
 * findMilkieDir (alfred's data-dir is not named `.milkie/`). Otherwise discover
 * the nearest `.milkie/` upward from cwd, preserving the original CLI behavior.
 */
function resolveTraceDir(dataDir: string | undefined): string {
  const dir = dataDir ? path.resolve(dataDir) : findMilkieDir(process.cwd())
  if (!dir) {
    throw new Error('no .milkie/ directory found upward from cwd (or pass --data-dir)')
  }
  return dir
}

/**
 * Build a Milkie instance with CLI defaults: persistent SQLite stateStore
 * (interrupt / resume survive across CLI processes), JsonlEventStore on
 * `.milkie/runs/`, and manifest auto-loaded.
 */
async function buildCliMilkie(): Promise<{ milkie: Milkie, milkieDir: string, stateStore: SQLiteStore, eventStore: JsonlEventStore }> {
  const milkieDir = findMilkieDir(process.cwd())
  if (!milkieDir) {
    throw new Error('no .milkie/ directory found upward from cwd')
  }
  const stateStore = new SQLiteStore({ path: path.join(milkieDir, 'state.sqlite') })
  await stateStore.init()
  const eventStore = new JsonlEventStore(path.join(milkieDir, 'runs'))
  const traceObjectStore = new FileTraceObjectStore(path.join(milkieDir, 'objects'))
  const milkie = new Milkie({ stateStore, eventStore, traceObjectStore })
  await milkie.loadManifest(path.join(milkieDir, 'agents.json'))
  return { milkie, milkieDir, stateStore, eventStore }
}

export interface MainResult {
  stdout:   string
  stderr:   string
  exitCode: number
}

/**
 * Programmatic entry to the milkie CLI. Returns captured stdout / stderr /
 * exit code instead of writing to process streams — makes the surface
 * unit-testable without spawning a child process.
 *
 * The real `bin` wrapper (src/cli/index.ts) is a thin shim that calls
 * `main(process.argv.slice(2))`, pipes the result to console, and exits.
 *
 * See `docs/superpowers/specs/2026-05-24-cli-surface-design.md` for the
 * verb surface this is implementing.
 */
export async function main(argv: string[]): Promise<MainResult> {
  const stdout: string[] = []
  const stderr: string[] = []
  let exitCode = 0

  const program = new Command()
  program
    .name('milkie')
    .exitOverride()   // commander throws CommanderError; we catch below
    .configureOutput({
      writeOut: (s) => { stdout.push(s) },
      writeErr: (s) => { stderr.push(s) },
    })

  const agent = program.command('agent')

  agent
    .command('list')
    .description('List registered agents (loaded from .milkie/agents.json)')
    .action(async () => {
      // ephemeral: only lists manifest agents, no persistent state needed.
      const milkie = new Milkie({ stateStore: new MemoryStore() })
      await milkie.loadManifest()
      for (const id of milkie.listAgents()) {
        stdout.push(JSON.stringify({ id, source: 'manifest' }) + '\n')
      }
    })

  agent
    .command('run <agentId>')
    .description('Execute an agent and record its run')
    .option('--input <text>',      'inline input')
    .option('--input-file <path>', 'read input from file')
    .option('--goal <text>',       'agent goal (defaults to input)')
    .option('--context-id <id>',   'context id for later resume / interrupt')
    .action(async (agentId: string, opts: { input?: string, inputFile?: string, goal?: string, contextId?: string }) => {
      const input = opts.input ?? (opts.inputFile ? fs.readFileSync(opts.inputFile, 'utf-8') : '')
      const goal  = opts.goal  ?? input
      const { milkie } = await buildCliMilkie()
      const result = await milkie.invoke({
        agentId,
        goal,
        input,
        contextId: opts.contextId,
      })
      stdout.push(JSON.stringify({
        runId:      result.agentRunId,
        contextId:  result.contextId,
        status:     result.status,
        lastOutput: result.output,
      }) + '\n')
    })

  agent
    .command('resume <contextId>')
    .description('Resume a paused agent from its latest checkpoint')
    .action(async (contextId: string) => {
      const { milkie, stateStore, eventStore } = await buildCliMilkie()
      const cpKey = `context:${contextId}:checkpoint:latest`
      // #73: resume state lives in the event log; resolve via context→runId pointer.
      const runId = await stateStore.get(`context:${contextId}:checkpoint-run:latest`) as string | undefined
      const checkpoint = runId ? checkpointFromEvents(await eventStore.readByRunId(runId)) : null
      if (!checkpoint) {
        throw new Error(`no checkpoint found for contextId "${contextId}"`)
      }
      const result = await milkie.resume(cpKey, checkpoint.meta.agentId, checkpoint.goal, '')
      stdout.push(JSON.stringify({
        runId:      result.agentRunId,
        contextId,
        status:     result.status,
        lastOutput: result.output,
      }) + '\n')
    })

  agent
    .command('interrupt <contextId>')
    .description('Signal a running agent to pause at its next turn boundary')
    .action(async (contextId: string) => {
      const { milkie } = await buildCliMilkie()
      await milkie.interrupt(contextId)
      stdout.push(JSON.stringify({ contextId, status: 'interrupt-signaled' }) + '\n')
    })

  const trace = program.command('trace')

  trace
    .command('inspect <runId>')
    .description('Print every event in a recorded run as JSONL')
    .option('--include-children', 'also emit events from descendant sub-agent runs')
    .option('--data-dir <path>', 'read trace from <path>/runs (e.g. a serve --data-dir); else find .milkie/ upward from cwd')
    .action(async (runId: string, opts: { includeChildren?: boolean; dataDir?: string }) => {
      const milkieDir = resolveTraceDir(opts.dataDir)
      const runsDir = path.join(milkieDir, 'runs')
      const eventStore = new JsonlEventStore(runsDir)

      const runIds = [runId]
      if (opts.includeChildren) {
        const { findDescendantRuns } = await import('../trace/render/children.js')
        runIds.push(...(await findDescendantRuns(runsDir, runId)))
      }
      for (const id of runIds) {
        for (const event of await eventStore.readByRunId(id)) {
          stdout.push(JSON.stringify(event) + '\n')
        }
      }
    })

  trace
    .command('render-html')
    .description('Render trace JSONL into a self-contained HTML report (reads --input file, writes HTML to stdout)')
    .requiredOption('--input <path>', 'JSONL file produced by `trace inspect` (or any equivalent source)')
    .action(async (opts: { input: string }) => {
      const { renderHtml } = await import('../trace/render/html.js')
      const content = fs.readFileSync(opts.input, 'utf-8')
      const events = content.split('\n')
        .filter(l => l.length > 0)
        .map(l => JSON.parse(l))
      stdout.push(renderHtml(events))
    })

  trace
    .command('report <runId>')
    .description('Render <runId> (and any descendant sub-agent runs) as a self-contained HTML report to stdout')
    .option('--data-dir <path>', 'read trace from <path>/runs (e.g. a serve --data-dir); else find .milkie/ upward from cwd')
    .action(async (runId: string, opts: { dataDir?: string }) => {
      const milkieDir = resolveTraceDir(opts.dataDir)
      const runsDir = path.join(milkieDir, 'runs')
      const eventStore = new JsonlEventStore(runsDir)
      const { findDescendantRuns } = await import('../trace/render/children.js')
      const { renderViewer } = await import('../trace/render/viewer.js')

      const runIds = [runId, ...(await findDescendantRuns(runsDir, runId))]
      const events = []
      for (const id of runIds) events.push(...(await eventStore.readByRunId(id)))

      const traceObjectStore = new FileTraceObjectStore(path.join(milkieDir, 'objects'))
      const regionContent = new Map<string, string>()
      for (const h of regionReuseCounts(events).keys()) {
        const c = await traceObjectStore.getCanonical(h)
        if (c !== undefined) regionContent.set(h, c)
      }

      stdout.push(renderViewer(events, { regionContent }))
    })

  trace
    .command('execution <runId>')
    .description('Project <runId> into execution-timeline JSON (steps with cache health + region composition) to stdout')
    .option('--data-dir <path>', 'read trace from <path>/runs (e.g. a serve --data-dir); else find .milkie/ upward from cwd')
    .action(async (runId: string, opts: { dataDir?: string }) => {
      const milkieDir = resolveTraceDir(opts.dataDir)
      const runsDir = path.join(milkieDir, 'runs')
      const eventStore = new JsonlEventStore(runsDir)
      const { buildExecutionProjection } = await import('../trace/diagnostics/buildExecutionProjection.js')

      const events = await eventStore.readByRunId(runId)

      const traceObjectStore = new FileTraceObjectStore(path.join(milkieDir, 'objects'))
      const regionContent = new Map<string, string>()
      for (const h of regionReuseCounts(events).keys()) {
        const c = await traceObjectStore.getCanonical(h)
        if (c !== undefined) regionContent.set(h, c)
      }

      stdout.push(JSON.stringify(buildExecutionProjection(events, { regionContent })) + '\n')
    })

  trace
    .command('replay <runId>')
    .description('Replay a recorded run from <data-dir or .milkie>/runs/<runId>.jsonl')
    .option('--data-dir <path>', 'read trace from <path>/runs (e.g. a serve --data-dir); else find .milkie/ upward from cwd')
    .action(async (runId: string, opts: { dataDir?: string }) => {
      const milkieDir = resolveTraceDir(opts.dataDir)
      const eventStore = new JsonlEventStore(path.join(milkieDir, 'runs'))
      const traceObjectStore = new FileTraceObjectStore(path.join(milkieDir, 'objects'))
      // ephemeral: replay is deterministic from the event log, no persistent state needed.
      const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore, traceObjectStore })
      await milkie.loadManifest(path.join(milkieDir, 'agents.json'))
      const result = await milkie.replay(runId)
      stdout.push(JSON.stringify({
        newRunId: runId,
        status:   result.status,
        output:   result.output,
      }) + '\n')
    })

  program
    .command('serve')
    .description('Run an HTTP + SSE server driving an agent (for external process integration, e.g. alfred). Foreground process; SIGTERM / closing stdin shuts it down.')
    .requiredOption('--agent <file>', 'agent definition file (.md with frontmatter)')
    .requiredOption('--port <port>', 'port to listen on (0 = OS-assigned)', v => parseInt(v, 10))
    .option('--host <host>', 'host/interface to bind', '127.0.0.1')
    .addOption(new Option('--state-store <kind>', 'persistence backend (#130: sqlite is restart-recoverable)').choices(['memory', 'sqlite']).default('memory'))
    .option('--data-dir <path>', 'stable directory for sqlite state + jsonl events (required with --state-store sqlite)')
    .action(async (opts: { agent: string; port: number; host: string; stateStore: 'memory' | 'sqlite'; dataDir?: string }) => {
      await serveMain({ agent: opts.agent, port: opts.port, host: opts.host, stateStore: opts.stateStore, dataDir: opts.dataDir })
    })

  try {
    await program.parseAsync(argv, { from: 'user' })
  } catch (err) {
    if (err && typeof err === 'object' && 'exitCode' in err) {
      // CommanderError from --help / --version / unknown command / bad arg.
      // commander already wrote help or error text through writeOut/writeErr.
      exitCode = (err as { exitCode: number }).exitCode
    } else {
      const msg = err instanceof Error ? err.message : String(err)
      stderr.push(JSON.stringify({ error: { code: 'CLI_ERROR', message: msg } }) + '\n')
      exitCode = 1
    }
  }

  return { stdout: stdout.join(''), stderr: stderr.join(''), exitCode }
}
