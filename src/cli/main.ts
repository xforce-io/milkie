import { Command } from 'commander'
import fs from 'fs'
import path from 'path'
import { Milkie } from '../runtime/Milkie.js'
import { JsonlEventStore } from '../trace/JsonlEventStore.js'

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
      const milkie = new Milkie()
      await milkie.loadManifest()
      for (const id of milkie.listAgents()) {
        stdout.push(JSON.stringify({ id, source: 'manifest' }) + '\n')
      }
    })

  const trace = program.command('trace')

  trace
    .command('replay <runId>')
    .description('Replay a recorded run from .milkie/runs/<runId>.jsonl')
    .action(async (runId: string) => {
      const milkieDir = findMilkieDir(process.cwd())
      if (!milkieDir) {
        throw new Error('no .milkie/ directory found upward from cwd')
      }
      const eventStore = new JsonlEventStore(path.join(milkieDir, 'runs'))
      const milkie = new Milkie({ eventStore })
      await milkie.loadManifest(path.join(milkieDir, 'agents.json'))
      const result = await milkie.replay(runId)
      stdout.push(JSON.stringify({
        newRunId: runId,
        status:   result.status,
        output:   result.output,
      }) + '\n')
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
