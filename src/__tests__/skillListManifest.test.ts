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

describe('skill_request manifest-backed fallback (#153)', () => {
  it('原生 AgentConfig 未声明但 manifest 有该 skill 时，不返回误导性的 unavailable', async () => {
    const p = writeManifest('request-fallback.json', JSON.stringify({
      skills: [
        { name: 'twitter-watch', description: '盯推', dir: '/abs/twitter-watch', version: '1.2.0' },
      ],
    }))
    process.env[ENV_KEY] = p
    const unavailableCtx = {
      requestSkill: (name: string) => ({ requested: name, status: 'unavailable' }),
    } as unknown as ToolContext

    const out = await skillRequest.handler({ name: 'twitter-watch' }, unavailableCtx) as Record<string, unknown>

    expect(out).toMatchObject({
      requested: 'twitter-watch',
      status:    'manifest_backed',
      skill:     { name: 'twitter-watch', description: '盯推', dir: '/abs/twitter-watch', version: '1.2.0' },
    })
    expect(out.status).not.toBe('unavailable')
    expect(out).toHaveProperty('instructionPath', '/abs/twitter-watch/SKILL.md')
    expect(String(out.message)).toContain('SKILL.md')
  })

  it('原生 skill_request 成功时保持原生返回，不走 manifest fallback', async () => {
    const p = writeManifest('native-wins.json', JSON.stringify({
      skills: [
        { name: 'twitter-watch', description: '盯推', dir: '/abs/twitter-watch', version: '1.2.0' },
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

    const out = await skillRequest.handler({ name: 'twitter-watch' }, nativeCtx)

    expect(out).toEqual({
      requested: 'twitter-watch',
      status:    'pending_next_epoch',
      version:   '9.9.9',
      scope:     'turn',
    })
  })
})
