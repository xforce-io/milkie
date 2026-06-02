import { main } from '../cli/main'
import { serveMain } from '../cli/serve'

describe('milkie serve — CLI wiring', () => {
  it('serve is a registered subcommand (listed in --help)', async () => {
    const res = await main(['--help'])
    expect(res.stdout).toContain('serve')
  })

  it('serve requires --agent and --port', async () => {
    const res = await main(['serve'])
    expect(res.exitCode).not.toBe(0)
    expect(res.stderr).toMatch(/required option|--agent|--port/i)
  })

  it('serveMain throws on a missing agent file (before it ever listens)', async () => {
    await expect(serveMain({ agent: '/no/such/agent.md', port: 0 })).rejects.toThrow()
  })
})
