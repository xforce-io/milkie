import { execFileSync } from 'node:child_process'
import { setTimeout as delay } from 'node:timers/promises'

export const EXECUTION_TOKEN_ENV = 'MILKIE_EXECUTION_TOKEN'
export interface OwnedProcess { pid: number; startedAt: string }
interface ProcessRow extends OwnedProcess { parent: number; state: string; tagged: boolean }

/** The snapshot never leaves this module: ps environment output may contain credentials. */
function snapshot(token: string): ProcessRow[] {
  const output = execFileSync('/bin/ps', ['eww', '-axo', 'pid=,ppid=,stat=,lstart=,command='], {
    encoding: 'utf8', timeout: 750, maxBuffer: 32 * 1024 * 1024,
    // Debian procps rejects -x unless the BSD personality is selected. macOS ps ignores this variable.
    env: { PATH: '/usr/bin:/bin', LC_ALL: 'C', PS_PERSONALITY: 'bsd' }, stdio: ['ignore', 'pipe', 'ignore'],
  })
  const marker = new RegExp(`(?:^|\\s)${EXECUTION_TOKEN_ENV}=${token}(?:\\s|$)`)
  const rows: ProcessRow[] = []
  for (const line of output.split('\n')) {
    const match = /^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+\s+\S+\s+\d+\s+\S+\s+\d+)\s+(.*)$/.exec(line)
    if (!match) continue
    rows.push({ pid: Number(match[1]), parent: Number(match[2]), state: match[3]!, startedAt: match[4]!, tagged: marker.test(match[5]!) })
  }
  if (!rows.some(row => row.pid === process.pid)) throw new Error('Process inventory unavailable.')
  return rows
}

/** Tracks inherited execution markers as well as observed descendants that later clear their environment. */
export class ProcessTracker {
  private readonly owned = new Map<number, string>()
  private readonly noted = new Set<number>()
  private timer?: NodeJS.Timeout
  private failed = false
  constructor(private readonly token: string, private readonly readInventory: (token: string) => ProcessRow[] = snapshot) {
    if (!/^[0-9a-f-]{36}$/.test(token)) throw new Error('Invalid process scope.')
    this.readInventory(token) // Refuse to launch if process inventory cannot be inspected.
  }
  /** Remember the direct child even when its environment is not visible in the process listing. */
  note(pid: number): void {
    if (!Number.isInteger(pid) || pid <= 0 || pid === process.pid) return
    this.noted.add(pid)
    if (!this.owned.has(pid)) this.owned.set(pid, '')
  }
  start(): void {
    this.timer = setInterval(() => { try { this.observe() } catch { this.failed = true } }, 200)
  }
  private observe(): ProcessRow[] {
    const rows = this.readInventory(this.token)
    const current = new Set<number>()
    for (const row of rows) {
      if (row.tagged || this.noted.has(row.pid) || this.owned.get(row.pid) === row.startedAt) current.add(row.pid)
    }
    let changed = true
    while (changed) {
      changed = false
      for (const row of rows) if (!current.has(row.pid) && current.has(row.parent)) { current.add(row.pid); changed = true }
    }
    const result = rows.filter(row => current.has(row.pid) && row.pid !== process.pid)
    for (const row of result) this.owned.set(row.pid, row.startedAt)
    return result.filter(row => !row.state.startsWith('Z'))
  }
  resources(): OwnedProcess[] { return [...this.owned].map(([pid, startedAt]) => ({ pid, startedAt })) }
  async stop(): Promise<boolean> {
    this.close()
    const started = Date.now(), deadline = started + 8250
    let emptyChecks = 0
    while (Date.now() < deadline) {
      let rows: ProcessRow[]
      try { rows = this.observe() } catch { this.failed = true; rows = [] }
      const liveNoted = [...this.noted].filter(pid => {
        try { process.kill(pid, 0); return true } catch { return false }
      })
      if (rows.length === 0 && liveNoted.length === 0) {
        if (++emptyChecks === 2) return !this.failed
      } else {
        emptyChecks = 0
        const signal = Date.now() - started < 1500 ? 'SIGTERM' : 'SIGKILL'
        // Each signal follows a fresh inventory: PID reuse cannot target an unrelated row.
        // Children first gives their CLI parent a chance to reap them.
        const pids = new Set<number>([...liveNoted, ...rows.map(row => row.pid)])
        for (const pid of [...pids].reverse()) {
          try { process.kill(pid, signal) } catch (e) {
            // A denied signal is not proof of liveness: subsequent inventories must still confirm exit.
            // If it remains live, the bounded stop returns false.
          }
        }
      }
      await delay(75)
    }
    return false
  }
  close(): void { if (this.timer) clearInterval(this.timer); this.timer = undefined }
}
