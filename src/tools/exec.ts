import { spawn } from 'node:child_process'
import type { ToolDefinition } from '../types/tool.js'

// Built-in shell/exec tool (#134). Lets an agent run subprocess commands —
// skill scripts (`python skills/.../scripts/x.py`), CLIs (`inv`, `codex-cli`), etc.
//
// Replay safety: this is an ordinary ToolDefinition. milkie records the handler's
// return on `tool.responded` and serves it from cache on replay (handler never
// re-runs — see ReplayingIOPort / ExplodingInnerPort). So the subprocess forks
// only at record time; replay reproduces the captured stdout/exit without side
// effects. No "non-replayable" contract is needed (mirrors create_plan).

const DEFAULT_TIMEOUT_MS = 120_000

// Per-stream capture cap. Capping inside the handler (rather than only via
// resultStrategy, which shapes the context view but not the event log) bounds
// BOTH the LLM context AND the persisted event log / CacheIndex. Head+tail is
// kept so the agent sees how the command started and ended.
const MAX_STREAM_CHARS = 30_000

export interface RunCommandInput {
  command:    string
  cwd?:       string
  timeoutMs?: number
}

export interface RunCommandOutput {
  stdout:    string
  stderr:    string
  exitCode:  number | null
  timedOut:  boolean
  truncated: boolean
}

function capStream(s: string): { text: string; truncated: boolean } {
  if (s.length <= MAX_STREAM_CHARS) return { text: s, truncated: false }
  const half    = Math.floor(MAX_STREAM_CHARS / 2)
  const dropped = s.length - MAX_STREAM_CHARS
  const text    = `${s.slice(0, half)}\n[...${dropped} chars dropped...]\n${s.slice(-half)}`
  return { text, truncated: true }
}

export function runCommand(input: RunCommandInput): Promise<RunCommandOutput> {
  const { command, cwd, timeoutMs = DEFAULT_TIMEOUT_MS } = input
  return new Promise((resolve) => {
    const child = spawn(command, { shell: true, cwd })
    let rawOut  = ''
    let rawErr  = ''
    let timedOut = false
    let settled  = false

    const timer = setTimeout(() => {
      timedOut = true
      child.kill('SIGKILL')
    }, timeoutMs)

    const finish = (exitCode: number | null, errSuffix = '') => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      const out = capStream(rawOut)
      const err = capStream(rawErr + errSuffix)
      resolve({
        stdout:    out.text,
        stderr:    err.text,
        exitCode,
        timedOut,
        truncated: out.truncated || err.truncated,
      })
    }

    child.stdout?.on('data', (d) => { rawOut += d.toString() })
    child.stderr?.on('data', (d) => { rawErr += d.toString() })
    // spawn failures (e.g. cwd does not exist) surface here, not via 'close'.
    child.on('error', (e: Error) => finish(null, (rawErr ? '\n' : '') + e.message))
    child.on('close', (code) => finish(code))
  })
}

export const execTools: ToolDefinition[] = [
  {
    name:        'run_command',
    description:
      'Execute a shell command in a subprocess and return { objectId?, stdout, stderr, exitCode, timedOut, truncated }. ' +
      'Use this to run skill scripts (e.g. `python skills/<name>/scripts/x.py`), CLIs (`inv`, `codex-cli`), ' +
      'and other tools. Output over 30000 chars per stream is truncated (head+tail kept). ' +
      'When stdout is non-empty the result carries an `objectId` for that output — to source a claim from this ' +
      'fetched data (a file you cat-ed, an API/DB you queried, a page you scraped), pass that objectId to the ' +
      'cite tool. Never write provenance like "(source:...)" in prose; cite the objectId instead.',
    inputSchema: {
      type:       'object',
      properties: {
        command:   { type: 'string', description: 'The shell command to run.' },
        cwd:       { type: 'string', description: 'Working directory for the command (optional).' },
        timeoutMs: { type: 'number', description: 'Timeout in ms (default 120000). On timeout the process is SIGKILLed and timedOut=true.' },
      },
      required:   ['command'],
    },
    handler: async (input, ctx) => {
      const out = await runCommand(input as RunCommandInput)
      // #148: lazy-register non-empty stdout as a citable object (like grep) so any
      // shell-fetched evidence (file / network / db) can be sourced via the `cite`
      // tool — no per-skill refactoring. registerObject is lazy (no event); the
      // object is promoted only when the agent actually cites it, so routine
      // commands (ls/mkdir) never flood the lineage graph. objectId is placed
      // FIRST so it survives the result-truncation strategy — the agent must see
      // it to cite. Stdout is a genuine tool-call record (origin invariant holds).
      if (ctx?.registerObject && out.stdout) {
        const { objectId } = ctx.registerObject({
          type: 'shell:stdout',
          meta: { command: (input as RunCommandInput).command, exitCode: out.exitCode },
        })
        return { objectId, ...out }
      }
      return out
    },
  },
]
