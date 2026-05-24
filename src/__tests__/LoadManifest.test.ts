import { Milkie } from '../runtime/Milkie'
import fs from 'fs'
import path from 'path'
import os from 'os'

describe('Milkie.loadManifest', () => {
  let tmpDir: string

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-manifest-'))
    fs.mkdirSync(path.join(tmpDir, '.milkie'))
    fs.mkdirSync(path.join(tmpDir, 'agents'))
  })

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true })
  })

  function writeAgentFile(name: string, agentId: string): void {
    const content = `---
agentId: ${agentId}
fsm:
  states: []
model:
  provider: stub
  model: stub
  adapter: stub
---
test prompt`
    fs.writeFileSync(path.join(tmpDir, 'agents', name), content)
  }

  function writeManifest(agents: { id: string; file: string }[]): string {
    const manifestPath = path.join(tmpDir, '.milkie', 'agents.json')
    fs.writeFileSync(manifestPath, JSON.stringify({ agents }))
    return manifestPath
  }

  it('loads all agents declared in the manifest', async () => {
    writeAgentFile('router.md', 'router')
    writeAgentFile('verifier.md', 'verifier')
    const manifestPath = writeManifest([
      { id: 'router',   file: '../agents/router.md' },
      { id: 'verifier', file: '../agents/verifier.md' },
    ])

    const milkie = new Milkie()
    const result = await milkie.loadManifest(manifestPath)

    expect(result.loaded).toEqual(['router', 'verifier'])
    expect(result.skipped).toEqual([])
    expect(milkie.getAgent('router')).toBeDefined()
    expect(milkie.getAgent('verifier')).toBeDefined()
  })

  it('skips entries whose agent file does not exist', async () => {
    writeAgentFile('router.md', 'router')
    // verifier.md intentionally NOT written
    const manifestPath = writeManifest([
      { id: 'router',   file: '../agents/router.md' },
      { id: 'verifier', file: '../agents/verifier.md' },
    ])

    const milkie = new Milkie()
    const result = await milkie.loadManifest(manifestPath)

    expect(result.loaded).toEqual(['router'])
    expect(result.skipped).toEqual([
      { id: 'verifier', reason: expect.stringMatching(/file not found|no such file/i) },
    ])
    expect(milkie.getAgent('verifier')).toBeUndefined()
  })

  it('skips entries whose frontmatter agentId does not match the manifest id', async () => {
    writeAgentFile('router.md', 'something-else')   // frontmatter says 'something-else'
    const manifestPath = writeManifest([
      { id: 'router', file: '../agents/router.md' },
    ])

    const milkie = new Milkie()
    const result = await milkie.loadManifest(manifestPath)

    expect(result.loaded).toEqual([])
    expect(result.skipped).toEqual([
      { id: 'router', reason: expect.stringMatching(/mismatch|agentId/i) },
    ])
    expect(milkie.getAgent('router')).toBeUndefined()
    expect(milkie.getAgent('something-else')).toBeUndefined()
  })

  it('throws when the manifest file itself is missing', async () => {
    const nonexistent = path.join(tmpDir, '.milkie', 'agents.json')   // never written
    const milkie = new Milkie()
    await expect(milkie.loadManifest(nonexistent)).rejects.toThrow(/no such file|not found/i)
  })

  it('with no path argument, searches upward from cwd for .milkie/agents.json', async () => {
    writeAgentFile('router.md', 'router')
    writeManifest([{ id: 'router', file: '../agents/router.md' }])

    // cwd is a sub-dir of tmpDir — upward walk should find tmpDir/.milkie/agents.json
    const subDir = fs.mkdtempSync(path.join(tmpDir, 'sub-'))
    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(subDir)
    try {
      const milkie = new Milkie()
      const result = await milkie.loadManifest()
      expect(result.loaded).toEqual(['router'])
    } finally {
      cwdSpy.mockRestore()
    }
  })

  it('returns empty loaded/skipped when no manifest is found upward from cwd', async () => {
    // beforeEach made tmpDir/.milkie but no agents.json inside it; nothing to find
    const cwdSpy = jest.spyOn(process, 'cwd').mockReturnValue(tmpDir)
    try {
      const milkie = new Milkie()
      const result = await milkie.loadManifest()
      expect(result.loaded).toEqual([])
      expect(result.skipped).toEqual([])
    } finally {
      cwdSpy.mockRestore()
    }
  })

  it('listAgents returns the ids of every registered agent', async () => {
    writeAgentFile('router.md',   'router')
    writeAgentFile('verifier.md', 'verifier')
    const manifestPath = writeManifest([
      { id: 'router',   file: '../agents/router.md' },
      { id: 'verifier', file: '../agents/verifier.md' },
    ])
    const milkie = new Milkie()
    await milkie.loadManifest(manifestPath)
    expect(milkie.listAgents().sort()).toEqual(['router', 'verifier'])
  })

  it('throws when the manifest is not valid JSON', async () => {
    const manifestPath = path.join(tmpDir, '.milkie', 'agents.json')
    fs.writeFileSync(manifestPath, '{ not valid json')
    const milkie = new Milkie()
    await expect(milkie.loadManifest(manifestPath)).rejects.toThrow()
  })

  it('skips duplicate ids — first wins, later occurrences reported with reason', async () => {
    writeAgentFile('router.md',     'router')
    writeAgentFile('router-v2.md',  'router')   // same agentId in frontmatter
    const manifestPath = writeManifest([
      { id: 'router', file: '../agents/router.md' },
      { id: 'router', file: '../agents/router-v2.md' },
    ])

    const milkie = new Milkie()
    const result = await milkie.loadManifest(manifestPath)

    expect(result.loaded).toEqual(['router'])
    expect(result.skipped).toEqual([
      { id: 'router', reason: expect.stringMatching(/duplicate/i) },
    ])
  })
})
