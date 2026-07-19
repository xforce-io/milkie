import fs from 'fs'
import path from 'path'
import type { ToolDefinition } from '../types/tool.js'
import { getLogger } from '../logging/logger.js'

// skill_list and skill_request are system-injected tools.
//
// #139 提议1 — skill_list 默认 handler 读宿主经 MILKIE_SKILL_MANIFEST 注入的本地
// manifest，返回真实完整技能列表。这是个 thin v1 bridge（非 §6 / s-010 完整 Skill
// Registry）：只读「宿主配置好的 manifest 文件路径」，不碰发现规则。设计取舍是
// 「最小契约（name+description）+ 开放透传」——milkie 只消费 name/description，宿主
// 附加字段（dir/version/…）原样透传给 LLM，不投影、不解释。
//
// 错误策略「行为软、日志硬」：未配置 / 读失败 / 单条目 malformed 一律 degrade 成安全
// 结果（绝不抛给 turn loop 里的 LLM），但已配置却读失败 / 跳过坏条目时 log WARNING，
// 避免 degrade 掩盖 misconfig。
const SKILL_MANIFEST_ENV = 'MILKIE_SKILL_MANIFEST'

interface SkillEntry { name: string; description: string; [k: string]: unknown }

function isValidSkill(s: unknown): s is SkillEntry {
  return !!s && typeof s === 'object'
    && typeof (s as Record<string, unknown>).name === 'string'
    && typeof (s as Record<string, unknown>).description === 'string'
}

function loadSkillManifest(): { skills: SkillEntry[]; registryConfigured: boolean } {
  const log = getLogger().child({ mod: 'tools' })
  const manifestPath = process.env[SKILL_MANIFEST_ENV]
  if (!manifestPath) return { skills: [], registryConfigured: false }

  let parsed: unknown
  try {
    parsed = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'))
  } catch (e) {
    // env 已设却读不到/解析失败 — 大概率 misconfig：行为 degrade，日志响亮。
    log.warn({ manifestPath, err: e as Error }, `skill_list: failed to read ${SKILL_MANIFEST_ENV}`)
    return { skills: [], registryConfigured: false }
  }

  // 顶层结构访问必须容错：JSON.parse('null') 返回 null、文件可能是数组/标量。
  // 任何取不出合法 skills 数组的情况都 degrade 成 registryConfigured:false（而非
  // true+空表）—— 后者会让 LLM 读成「registry 说我零技能」，正是 #139 要消灭的误导性空。
  const skillsRaw = (parsed && typeof parsed === 'object' && !Array.isArray(parsed))
    ? (parsed as { skills?: unknown }).skills
    : undefined
  if (!Array.isArray(skillsRaw)) {
    log.warn({ manifestPath }, `skill_list: ${SKILL_MANIFEST_ENV} parsed but has no valid 'skills' array; treating as unconfigured`)
    return { skills: [], registryConfigured: false }
  }

  const skills: SkillEntry[] = []
  for (const s of skillsRaw) {
    if (isValidSkill(s)) skills.push(s)               // 原样透传，含 dir/version 等附加字段
    else log.warn({ entry: JSON.stringify(s) }, 'skill_list: skipping malformed skill entry (missing name/description)')
  }
  return { skills, registryConfigured: true }
}

function normalizeSkillName(name: string): string {
  return name.trim().replace(/\s+skill$/i, '')
}

/** Max chars of SKILL.md body returned in tool_result (UTF-16 code units / JS string.length). */
export const SKILL_INSTRUCTIONS_MAX_CHARS = 16_000

/**
 * Closed-world skill load via host manifest (#164).
 *
 * Hit: read only manifest.dir/SKILL.md and return instructions + dir.
 * Miss: not_found without filesystem search.
 * Registry unconfigured: undefined so the handler keeps native unavailable.
 */
function manifestBackedSkillRequest(name: string): Record<string, unknown> | undefined {
  const manifest = loadSkillManifest()
  if (!manifest.registryConfigured) return undefined

  const normalized = normalizeSkillName(name)
  const skill = manifest.skills.find(s => normalizeSkillName(s.name) === normalized)
  if (!skill) {
    return {
      requested: normalized,
      status:    'not_found',
      message:   `Skill "${normalized}" is not in the skill manifest.`,
    }
  }

  const dirRaw = typeof skill.dir === 'string' ? skill.dir : undefined
  const dir = dirRaw?.trim() ? dirRaw : undefined
  if (!dir) {
    return {
      requested: normalized,
      status:    'missing_dir',
      skill,
      message:   `Skill "${normalized}" is in the skill manifest but has no dir field; cannot load SKILL.md.`,
    }
  }
  // Authoritative dir must be absolute; never resolve relative paths against cwd.
  if (!path.isAbsolute(dir)) {
    return {
      requested: normalized,
      status:    'invalid_dir',
      skill,
      dir,
      message:   `Skill "${normalized}" dir must be an absolute path; got ${JSON.stringify(dir)}. Do not search the filesystem for SKILL.md.`,
    }
  }

  const instructionPath = path.join(dir, 'SKILL.md')
  // Closed-world: read only this absolute path; never search other directories.
  let text: string
  try {
    text = fs.readFileSync(instructionPath, 'utf-8')
  } catch {
    return {
      requested: normalized,
      status:    'read_error',
      skill,
      dir,
      instructionPath,
      message:   `Failed to read skill instructions at ${instructionPath}. Do not search the filesystem for SKILL.md.`,
    }
  }

  const truncated = text.length > SKILL_INSTRUCTIONS_MAX_CHARS
  const instructions = truncated ? text.slice(0, SKILL_INSTRUCTIONS_MAX_CHARS) : text
  return {
    requested: normalized,
    status:    'ok',
    skill,
    dir,
    instructionPath,
    instructions,
    truncated,
    ...(truncated
      ? {
          message:
            `Instructions truncated to ${SKILL_INSTRUCTIONS_MAX_CHARS} chars. ` +
            `Continue reading only at instructionPath (known absolute path); do not search the filesystem.`,
        }
      : {}),
  }
}

export const systemTools: ToolDefinition[] = [
  {
    name:        'skill_list',
    description: 'List all available skills with their names and brief descriptions.',
    inputSchema: { type: 'object', properties: {}, required: [] },
    handler: async (_input, _ctx) => {
      return loadSkillManifest()
    },
  },

  {
    name:        'skill_request',
    description:
      'Load skill instructions by name from the host skill manifest (closed-world). ' +
      'On success returns instructions text and authoritative dir. Prefer this over shell find/cat to discover SKILL.md. ' +
      'Optional scope applies when the skill is also declared in AgentConfig.skillInstructions (next context epoch). ' +
      "Choose scope=turn for one-shot usage (auto-released at turn end) or scope=session for cross-turn persistence. Default: turn.",
    inputSchema: {
      type:       'object',
      properties: {
        name:  { type: 'string', description: 'Skill name to load' },
        scope: {
          type: 'string',
          enum: ['turn', 'session'],
          description: "Lifetime: 'turn' = available only for the current conversational turn (auto-released at turn end); 'session' = persists across turns. Default: 'turn'.",
        },
      },
      required:   ['name'],
    },
    handler: async (input: unknown, ctx) => {
      const { name, scope } = input as { name: string; scope?: 'turn' | 'session' }
      const result = ctx.requestSkill?.(name, scope) ?? { requested: name, status: 'unavailable' }
      if ((result as { status?: unknown }).status !== 'unavailable') return result
      return manifestBackedSkillRequest(name) ?? result
    },
  },
]
