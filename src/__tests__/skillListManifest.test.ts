import fs from 'fs'
import os from 'os'
import path from 'path'
import { systemTools } from '../tools/system'
import type { ToolContext } from '../types/tool'
import { createServiceLogger, setLogger } from '../logging/logger'

// #139 提议1: skill_list 默认 handler 读 MILKIE_SKILL_MANIFEST 指向的本地 manifest
// → 返回真实完整技能列表；未配置 / 读失败 → degrade（行为软）+ WARNING（日志硬）。

const skillList = systemTools.find(t => t.name === 'skill_list')!
const skillRequest = systemTools.find(t => t.name === 'skill_request')!
const ctx = {} as unknown as ToolContext

let tmpDir: string
const ENV_KEY = 'MILKIE_SKILL_MANIFEST'
let savedEnv: string | undefined
let logRaw: string[]
/** #79：WARNING 断言走注入的服务日志（mod=tools），不再 spy console。 */
const warnLines = (): Record<string, unknown>[] =>
  logRaw.flatMap(s => s.split('\n').filter(Boolean))
    .map(s => JSON.parse(s) as Record<string, unknown>)
    .filter(l => l.level === 'warn')

beforeAll(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-skill-manifest-'))
})
afterAll(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true })
})
beforeEach(() => {
  savedEnv = process.env[ENV_KEY]
  delete process.env[ENV_KEY]
  logRaw = []
  setLogger(createServiceLogger({
    level: 'warn', format: 'json',
    destination: { write: (s: string) => { logRaw.push(s) } },
  }))
})
afterEach(() => {
  if (savedEnv === undefined) delete process.env[ENV_KEY]
  else process.env[ENV_KEY] = savedEnv
  setLogger(undefined)
})

function writeManifest(name: string, content: string): string {
  const p = path.join(tmpDir, name)
  fs.writeFileSync(p, content, 'utf-8')
  return p
}

describe('skill_list 默认 handler 读 manifest (#139)', () => {
  it('env 未设 → degrade 安静：{skills:[], registryConfigured:false}，不 WARNING', async () => {
    const out = await skillList.handler({}, ctx) as { skills: unknown[]; registryConfigured: boolean }
    expect(out.skills).toEqual([])
    expect(out.registryConfigured).toBe(false)
    expect(warnLines()).toHaveLength(0)
  })

  it('env 已设、manifest 有效 → 返回完整列表，原样透传宿主附加字段（dir/version 不投影）', async () => {
    const p = writeManifest('ok.json', JSON.stringify({
      skills: [
        { name: 'twitter-watch', description: '盯推', dir: '/abs/twitter-watch', version: '1.2.0' },
        { name: 'agent-docs-qa', description: '文档问答', dir: '/abs/agent-docs-qa' },
      ],
    }))
    process.env[ENV_KEY] = p
    const out = await skillList.handler({}, ctx) as { skills: Array<Record<string, unknown>>; registryConfigured: boolean }
    expect(out.registryConfigured).toBe(true)
    expect(out.skills).toHaveLength(2)
    expect(out.skills[0]).toEqual({ name: 'twitter-watch', description: '盯推', dir: '/abs/twitter-watch', version: '1.2.0' })
    expect(out.skills[1]).toEqual({ name: 'agent-docs-qa', description: '文档问答', dir: '/abs/agent-docs-qa' })
  })

  it('env 已设、文件缺失 → degrade {skills:[], registryConfigured:false} + WARNING', async () => {
    process.env[ENV_KEY] = path.join(tmpDir, 'does-not-exist.json')
    const out = await skillList.handler({}, ctx) as { skills: unknown[]; registryConfigured: boolean }
    expect(out.skills).toEqual([])
    expect(out.registryConfigured).toBe(false)
    expect(warnLines().length).toBeGreaterThan(0)
    expect(warnLines()[0]!.mod).toBe('tools')
    expect(warnLines()[0]!.level).toBe('warn')
  })

  it('env 已设、JSON 损坏 → degrade + WARNING', async () => {
    const p = writeManifest('broken.json', '{ not valid json')
    process.env[ENV_KEY] = p
    const out = await skillList.handler({}, ctx) as { skills: unknown[]; registryConfigured: boolean }
    expect(out.skills).toEqual([])
    expect(out.registryConfigured).toBe(false)
    expect(warnLines().length).toBeGreaterThan(0)
  })

  it('合法 JSON 但顶层为 null → 不抛、degrade false + WARNING（契约点2：绝不抛给 LLM）', async () => {
    const p = writeManifest('null.json', 'null')
    process.env[ENV_KEY] = p
    // 关键：handler 必须 resolve（不能 reject/throw），否则会成为 tool-call error 丢给 turn loop
    const out = await skillList.handler({}, ctx) as { skills: unknown[]; registryConfigured: boolean }
    expect(out.skills).toEqual([])
    expect(out.registryConfigured).toBe(false)
    expect(warnLines().length).toBeGreaterThan(0)
  })

  it('合法 JSON 但缺 skills 键（{}）→ degrade false + WARNING（不静默 true 空表，避免重新引入误导性空）', async () => {
    const p = writeManifest('noskills.json', JSON.stringify({}))
    process.env[ENV_KEY] = p
    const out = await skillList.handler({}, ctx) as { skills: unknown[]; registryConfigured: boolean }
    expect(out.skills).toEqual([])
    expect(out.registryConfigured).toBe(false)
    expect(warnLines().length).toBeGreaterThan(0)
  })

  it('skills 非数组（{"skills":"x"}）→ degrade false + WARNING', async () => {
    const p = writeManifest('nonarray.json', JSON.stringify({ skills: 'x' }))
    process.env[ENV_KEY] = p
    const out = await skillList.handler({}, ctx) as { skills: unknown[]; registryConfigured: boolean }
    expect(out.skills).toEqual([])
    expect(out.registryConfigured).toBe(false)
    expect(warnLines().length).toBeGreaterThan(0)
  })

  it('合法空数组（{"skills":[]}）→ registryConfigured:true，宿主显式声明零技能，不 WARNING', async () => {
    const p = writeManifest('empty.json', JSON.stringify({ skills: [] }))
    process.env[ENV_KEY] = p
    const out = await skillList.handler({}, ctx) as { skills: unknown[]; registryConfigured: boolean }
    expect(out.skills).toEqual([])
    expect(out.registryConfigured).toBe(true)
    expect(warnLines()).toHaveLength(0)
  })

  it('单条目 malformed（缺 name/description）→ 跳过该条 + WARNING，其余正常返回', async () => {
    const p = writeManifest('partial.json', JSON.stringify({
      skills: [
        { name: 'good', description: '有效' },
        { name: 'no-desc' },                       // 缺 description → 跳过
        { description: 'no-name' },                // 缺 name → 跳过
        { name: 'also-good', description: '也有效', dir: '/abs/x' },
      ],
    }))
    process.env[ENV_KEY] = p
    const out = await skillList.handler({}, ctx) as { skills: Array<Record<string, unknown>>; registryConfigured: boolean }
    expect(out.registryConfigured).toBe(true)
    expect(out.skills.map(s => s.name)).toEqual(['good', 'also-good'])
    expect(warnLines().length).toBeGreaterThan(0)
  })
})

const unavailableCtx = {
  requestSkill: (name: string) => ({ requested: name, status: 'unavailable' }),
} as unknown as ToolContext

/** Create a real skill dir with SKILL.md and return its absolute path. */
function makeSkillDir(name: string, body: string): string {
  const dir = path.join(tmpDir, 'skills', name)
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'SKILL.md'), body, 'utf-8')
  return dir
}

describe('skill_request closed-world manifest load (#164)', () => {
  // M1: hit + real SKILL.md → status ok, instructions + dir; no run_command/cat load guidance
  it('M1: manifest hit returns instructions + dir (not cat/find guidance)', async () => {
    const dir = makeSkillDir('web', '# Web\n\nFetch and analyze web pages.\n')
    const p = writeManifest('request-hit.json', JSON.stringify({
      skills: [
        { name: 'web', description: 'web skill', dir, version: '1.0.0' },
      ],
    }))
    process.env[ENV_KEY] = p

    const out = await skillRequest.handler({ name: 'web' }, unavailableCtx) as Record<string, unknown>

    expect(out).toMatchObject({
      requested: 'web',
      status:    'ok',
      dir,
      instructionPath: path.join(dir, 'SKILL.md'),
      truncated: false,
    })
    expect(String(out.instructions)).toContain('Fetch and analyze web pages')
    expect(out.skill).toMatchObject({ name: 'web', description: 'web skill', dir })
    // Must not guide shell discovery as the primary load path
    const msg = out.message != null ? String(out.message) : ''
    expect(msg).not.toMatch(/run_command\/cat/i)
    expect(msg).not.toMatch(/Read .+ with run_command/i)
    expect(JSON.stringify(out)).not.toMatch(/Read .+ with run_command\/cat/i)
  })

  // M2: miss when registry configured → not_found; no disk search outside known paths
  it('M2: unknown skill name returns not_found without disk scan', async () => {
    const dir = makeSkillDir('web', '# Web\n\nbody\n')
    const skillMd = path.join(dir, 'SKILL.md')
    const p = writeManifest('request-miss.json', JSON.stringify({
      skills: [{ name: 'web', description: 'web skill', dir }],
    }))
    process.env[ENV_KEY] = p

    const readSpy = jest.spyOn(fs, 'readFileSync')
    try {
      const out = await skillRequest.handler({ name: 'no-such-skill' }, unavailableCtx) as Record<string, unknown>

      expect(out).toMatchObject({
        requested: 'no-such-skill',
        status:    'not_found',
      })
      expect(out).not.toHaveProperty('instructions')
      expect(String(out.message ?? '')).not.toMatch(/find|run_command\/cat|search/i)
      // Miss must not open any skill dir path (manifest JSON read is OK)
      const skillPathsRead = readSpy.mock.calls
        .map(c => String(c[0]))
        .filter(f => f === skillMd || f.startsWith(dir + path.sep) || f === dir)
      expect(skillPathsRead).toEqual([])
    } finally {
      readSpy.mockRestore()
    }
  })

  // M3: registry not configured → unavailable (degrade)
  it('M3: registry unconfigured stays unavailable without throw', async () => {
    // ENV_KEY deleted in beforeEach
    const out = await skillRequest.handler({ name: 'web' }, unavailableCtx) as Record<string, unknown>
    expect(out).toMatchObject({ requested: 'web', status: 'unavailable' })
  })

  // M4: native pending_next_epoch wins; no disk read for manifest path
  it('M4: native pending_next_epoch wins over manifest path', async () => {
    const dir = makeSkillDir('twitter-watch', '# Twitter\n\nwatch\n')
    const skillMd = path.join(dir, 'SKILL.md')
    const p = writeManifest('native-wins.json', JSON.stringify({
      skills: [
        { name: 'twitter-watch', description: '盯推', dir, version: '1.2.0' },
      ],
    }))
    process.env[ENV_KEY] = p
    const nativeCtx = {
      requestSkill: (_name: string) => ({
        requested: 'twitter-watch',
        status:    'pending_next_epoch',
        version:   '9.9.9',
        scope:     'turn',
      }),
    } as unknown as ToolContext

    const readSpy = jest.spyOn(fs, 'readFileSync')
    try {
      const out = await skillRequest.handler({ name: 'twitter-watch' }, nativeCtx)

      expect(out).toEqual({
        requested: 'twitter-watch',
        status:    'pending_next_epoch',
        version:   '9.9.9',
        scope:     'turn',
      })
      // Native hit must not touch skill SKILL.md (or any skill dir file)
      const skillPathsRead = readSpy.mock.calls
        .map(c => String(c[0]))
        .filter(f => f === skillMd || f.startsWith(dir + path.sep) || f === dir)
      expect(skillPathsRead).toEqual([])
    } finally {
      readSpy.mockRestore()
    }
  })

  // M5: truncation at 16k
  it('M5: large SKILL.md is truncated to 16000 chars with truncated=true', async () => {
    const body = 'X'.repeat(20_000)
    const dir = makeSkillDir('big', body)
    const p = writeManifest('request-trunc.json', JSON.stringify({
      skills: [{ name: 'big', description: 'big skill', dir }],
    }))
    process.env[ENV_KEY] = p

    const out = await skillRequest.handler({ name: 'big' }, unavailableCtx) as Record<string, unknown>

    expect(out.status).toBe('ok')
    expect(out.truncated).toBe(true)
    expect(String(out.instructions)).toHaveLength(16_000)
    expect(out.dir).toBe(dir)
    expect(out.instructionPath).toBe(path.join(dir, 'SKILL.md'))
    // Truncation message may mention known absolute path; must not suggest search
    if (out.message != null) {
      expect(String(out.message)).not.toMatch(/find \$HOME|search the filesystem to discover/i)
      expect(String(out.message)).toMatch(/instructionPath|truncated/i)
    }
  })

  // M6: missing dir / missing SKILL.md
  it('M6a: skill entry without dir returns missing_dir', async () => {
    const p = writeManifest('no-dir.json', JSON.stringify({
      skills: [{ name: 'ghost', description: 'no dir field' }],
    }))
    process.env[ENV_KEY] = p

    const out = await skillRequest.handler({ name: 'ghost' }, unavailableCtx) as Record<string, unknown>

    expect(out).toMatchObject({
      requested: 'ghost',
      status:    'missing_dir',
    })
    expect(out).not.toHaveProperty('instructions')
    expect(String(out.message ?? '')).not.toMatch(/run_command\/cat|find \$HOME/i)
  })

  it('M6a2: empty dir string returns missing_dir without reading disk', async () => {
    const p = writeManifest('empty-dir.json', JSON.stringify({
      skills: [{ name: 'ghost', description: 'empty dir', dir: '   ' }],
    }))
    process.env[ENV_KEY] = p
    const readSpy = jest.spyOn(fs, 'readFileSync')
    try {
      const out = await skillRequest.handler({ name: 'ghost' }, unavailableCtx) as Record<string, unknown>
      expect(out).toMatchObject({ requested: 'ghost', status: 'missing_dir' })
      // Only the manifest JSON path may be read
      const nonManifestReads = readSpy.mock.calls
        .map(c => String(c[0]))
        .filter(f => f !== p)
      expect(nonManifestReads).toEqual([])
    } finally {
      readSpy.mockRestore()
    }
  })

  it('M6a3: relative dir returns invalid_dir without reading disk', async () => {
    const p = writeManifest('rel-dir.json', JSON.stringify({
      skills: [{ name: 'rel', description: 'relative', dir: 'skills/web' }],
    }))
    process.env[ENV_KEY] = p
    const readSpy = jest.spyOn(fs, 'readFileSync')
    try {
      const out = await skillRequest.handler({ name: 'rel' }, unavailableCtx) as Record<string, unknown>
      expect(out).toMatchObject({
        requested: 'rel',
        status:    'invalid_dir',
        dir:       'skills/web',
      })
      expect(out).not.toHaveProperty('instructions')
      const nonManifestReads = readSpy.mock.calls
        .map(c => String(c[0]))
        .filter(f => f !== p)
      expect(nonManifestReads).toEqual([])
    } finally {
      readSpy.mockRestore()
    }
  })

  it('M6b: dir present but SKILL.md missing returns read_error', async () => {
    const dir = path.join(tmpDir, 'skills', 'broken-md')
    fs.mkdirSync(dir, { recursive: true })
    // deliberately no SKILL.md
    const p = writeManifest('read-err.json', JSON.stringify({
      skills: [{ name: 'broken-md', description: 'missing md', dir }],
    }))
    process.env[ENV_KEY] = p

    const out = await skillRequest.handler({ name: 'broken-md' }, unavailableCtx) as Record<string, unknown>

    expect(out).toMatchObject({
      requested: 'broken-md',
      status:    'read_error',
      dir,
      instructionPath: path.join(dir, 'SKILL.md'),
    })
    expect(out).not.toHaveProperty('instructions')
    expect(String(out.message ?? '')).not.toMatch(/run_command\/cat|find \$HOME/i)
  })

  // M7: no regression to "Read … with run_command/cat" as hit primary path
  it('M7: hit path must not use status manifest_backed or cat load guidance', async () => {
    const dir = makeSkillDir('ops', '# Ops\n\nops body\n')
    const p = writeManifest('no-regress.json', JSON.stringify({
      skills: [{ name: 'ops', description: 'ops', dir }],
    }))
    process.env[ENV_KEY] = p

    const out = await skillRequest.handler({ name: 'ops' }, unavailableCtx) as Record<string, unknown>

    expect(out.status).toBe('ok')
    expect(out.status).not.toBe('manifest_backed')
    expect(JSON.stringify(out)).not.toContain('Read ')
    expect(JSON.stringify(out)).not.toMatch(/run_command\/cat/)
  })

  it('normalizes trailing " skill" suffix on request name', async () => {
    const dir = makeSkillDir('web', '# Web\n\nbody\n')
    const p = writeManifest('norm.json', JSON.stringify({
      skills: [{ name: 'web', description: 'web', dir }],
    }))
    process.env[ENV_KEY] = p

    const out = await skillRequest.handler({ name: 'web skill' }, unavailableCtx) as Record<string, unknown>
    expect(out).toMatchObject({ requested: 'web', status: 'ok' })
  })
})
