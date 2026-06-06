import { mkdtemp, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { execTools, runCommand, type RunCommandOutput } from '../tools/exec'
import type { ToolContext } from '../types/tool'

const runCmd = execTools.find(t => t.name === 'run_command')!
const NODE = process.execPath

describe('built-in run_command tool (#134)', () => {
  it('registers run_command', () => {
    expect(runCmd).toBeTruthy()
    expect(runCmd.name).toBe('run_command')
  })

  it('runs a command and captures stdout + exitCode 0', async () => {
    const out = await runCommand({ command: 'echo hello-milkie' })
    expect(out.stdout).toContain('hello-milkie')
    expect(out.exitCode).toBe(0)
    expect(out.timedOut).toBe(false)
    expect(out.truncated).toBe(false)
  })

  it('propagates a non-zero exit code', async () => {
    const out = await runCommand({ command: 'exit 3' })
    expect(out.exitCode).toBe(3)
    expect(out.timedOut).toBe(false)
  })

  it('captures stderr separately from stdout', async () => {
    const out = await runCommand({ command: 'echo oops 1>&2' })
    expect(out.stderr).toContain('oops')
    expect(out.stdout).toBe('')
    expect(out.exitCode).toBe(0)
  })

  it('honors cwd', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'milkie-exec-'))
    try {
      await writeFile(join(dir, 'marker.txt'), 'present')
      const out = await runCommand({ command: 'cat marker.txt', cwd: dir })
      expect(out.stdout).toContain('present')
      expect(out.exitCode).toBe(0)
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  it('kills the process on timeout and reports timedOut', async () => {
    const start = Date.now()
    const out = await runCommand({ command: `${NODE} -e "setTimeout(()=>{}, 10000)"`, timeoutMs: 200 })
    expect(out.timedOut).toBe(true)
    expect(out.exitCode).not.toBe(0) // SIGKILLed → null or non-zero, never clean 0
    expect(Date.now() - start).toBeLessThan(5000) // proves it was killed, not waited out
  })

  it('surfaces spawn errors (bad cwd) without throwing', async () => {
    const out = await runCommand({ command: 'echo x', cwd: '/no/such/dir/at/all' })
    expect(out.exitCode).toBeNull()
    expect(out.stderr.length).toBeGreaterThan(0)
  })

  it('truncates very large output (head+tail kept, both ends visible)', async () => {
    const out = await runCommand({
      command: `${NODE} -e "process.stdout.write('A'.repeat(50)+'B'.repeat(100000)+'Z'.repeat(50))"`,
    })
    expect(out.truncated).toBe(true)
    expect(out.stdout.length).toBeLessThan(40_000)
    expect(out.stdout.startsWith('AAAAA')).toBe(true)   // head kept
    expect(out.stdout.endsWith('ZZZZZ')).toBe(true)     // tail kept
    expect(out.stdout).toContain('chars dropped')
  })

  // #148: lazy-register stdout as a citable object so any shell-fetched evidence
  // (file/network/db) can be cited via the framework `cite` tool.
  it('lazy-registers a shell:stdout object for non-empty stdout and surfaces objectId first', async () => {
    const registered: Array<{ type: string; meta?: Record<string, unknown> }> = []
    const created: unknown[] = []
    const ctx = {
      registerObject: (spec: { type: string; meta?: Record<string, unknown> }) => { registered.push(spec); return { objectId: `obj:reg:${registered.length}` } },
      createObject:   (spec: unknown) => { created.push(spec); return { objectId: 'obj:eager' } },
    } as unknown as ToolContext
    const out = (await runCmd.handler({ command: 'echo evidence-xyz' }, ctx)) as RunCommandOutput & { objectId?: string }
    expect(out.objectId).toBe('obj:reg:1')
    expect(registered).toHaveLength(1)
    expect(registered[0]!.type).toBe('shell:stdout')
    expect(created).toHaveLength(0) // lazy: registerObject, not eager createObject
    expect(out.stdout).toContain('evidence-xyz')
  })

  it('does not register an object when stdout is empty (e.g. stderr-only)', async () => {
    const registered: unknown[] = []
    const ctx = {
      registerObject: (spec: unknown) => { registered.push(spec); return { objectId: 'obj:x' } },
    } as unknown as ToolContext
    const out = (await runCmd.handler({ command: 'echo oops 1>&2' }, ctx)) as RunCommandOutput & { objectId?: string }
    expect(registered).toHaveLength(0)
    expect(out.objectId).toBeUndefined()
  })

  it('handler is invokable as a ToolDefinition (ctx not required)', async () => {
    const out = (await runCmd.handler({ command: 'echo viahandler' }, {} as ToolContext)) as RunCommandOutput
    expect(out.stdout).toContain('viahandler')
  })
})
