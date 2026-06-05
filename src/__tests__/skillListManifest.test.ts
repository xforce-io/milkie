import fs from 'fs'
import os from 'os'
import path from 'path'
import { systemTools } from '../tools/system'
import type { ToolContext } from '../types/tool'

// #139 提议1: skill_list 默认 handler 读 MILKIE_SKILL_MANIFEST 指向的本地 manifest
// → 返回真实完整技能列表；未配置 / 读失败 → degrade（行为软）+ WARNING（日志硬）。

const skillList = systemTools.find(t => t.name === 'skill_list')!
const ctx = {} as unknown as ToolContext

let tmpDir: string
const ENV_KEY = 'MILKIE_SKILL_MANIFEST'
let savedEnv: string | undefined
let warnSpy: jest.SpyInstance

beforeAll(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-skill-manifest-'))
})
afterAll(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true })
})
beforeEach(() => {
  savedEnv = process.env[ENV_KEY]
  delete process.env[ENV_KEY]
  warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {})
})
afterEach(() => {
  if (savedEnv === undefined) delete process.env[ENV_KEY]
  else process.env[ENV_KEY] = savedEnv
  warnSpy.mockRestore()
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
    expect(warnSpy).not.toHaveBeenCalled()
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
    expect(warnSpy).toHaveBeenCalled()
  })

  it('env 已设、JSON 损坏 → degrade + WARNING', async () => {
    const p = writeManifest('broken.json', '{ not valid json')
    process.env[ENV_KEY] = p
    const out = await skillList.handler({}, ctx) as { skills: unknown[]; registryConfigured: boolean }
    expect(out.skills).toEqual([])
    expect(out.registryConfigured).toBe(false)
    expect(warnSpy).toHaveBeenCalled()
  })

  it('合法 JSON 但顶层为 null → 不抛、degrade false + WARNING（契约点2：绝不抛给 LLM）', async () => {
    const p = writeManifest('null.json', 'null')
    process.env[ENV_KEY] = p
    // 关键：handler 必须 resolve（不能 reject/throw），否则会成为 tool-call error 丢给 turn loop
    const out = await skillList.handler({}, ctx) as { skills: unknown[]; registryConfigured: boolean }
    expect(out.skills).toEqual([])
    expect(out.registryConfigured).toBe(false)
    expect(warnSpy).toHaveBeenCalled()
  })

  it('合法 JSON 但缺 skills 键（{}）→ degrade false + WARNING（不静默 true 空表，避免重新引入误导性空）', async () => {
    const p = writeManifest('noskills.json', JSON.stringify({}))
    process.env[ENV_KEY] = p
    const out = await skillList.handler({}, ctx) as { skills: unknown[]; registryConfigured: boolean }
    expect(out.skills).toEqual([])
    expect(out.registryConfigured).toBe(false)
    expect(warnSpy).toHaveBeenCalled()
  })

  it('skills 非数组（{"skills":"x"}）→ degrade false + WARNING', async () => {
    const p = writeManifest('nonarray.json', JSON.stringify({ skills: 'x' }))
    process.env[ENV_KEY] = p
    const out = await skillList.handler({}, ctx) as { skills: unknown[]; registryConfigured: boolean }
    expect(out.skills).toEqual([])
    expect(out.registryConfigured).toBe(false)
    expect(warnSpy).toHaveBeenCalled()
  })

  it('合法空数组（{"skills":[]}）→ registryConfigured:true，宿主显式声明零技能，不 WARNING', async () => {
    const p = writeManifest('empty.json', JSON.stringify({ skills: [] }))
    process.env[ENV_KEY] = p
    const out = await skillList.handler({}, ctx) as { skills: unknown[]; registryConfigured: boolean }
    expect(out.skills).toEqual([])
    expect(out.registryConfigured).toBe(true)
    expect(warnSpy).not.toHaveBeenCalled()
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
    expect(warnSpy).toHaveBeenCalled()
  })
})
