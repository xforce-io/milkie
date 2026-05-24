import { Command } from 'commander'
import { Milkie } from '../runtime/Milkie.js'

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
    .exitOverride((err) => { exitCode = err.exitCode })
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

  try {
    await program.parseAsync(argv, { from: 'user' })
  } catch (err) {
    if (exitCode === 0) {
      const msg = err instanceof Error ? err.message : String(err)
      stderr.push(JSON.stringify({ error: { code: 'CLI_ERROR', message: msg } }) + '\n')
      exitCode = 1
    }
  }

  return { stdout: stdout.join(''), stderr: stderr.join(''), exitCode }
}
