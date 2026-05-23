/**
 * Case 5: Skill 渐进加载与 A/B 版本对比
 *
 * 验证：instructions bucket 动态加载、contextEpoch 递增、
 * 两个 Agent 版本的 A/B 输出对比（skill 指令差异导致行为差异）
 */

import type { AgentConfig } from '../../src/types/agent.js'
import type { AgentResult } from '../../src/types/common.js'
import type { Trajectory } from '../../src/types/trajectory.js'
import type { ToolDefinition } from '../../src/types/tool.js'
import { Milkie } from '../../src/runtime/Milkie.js'
import { TrajectoryStore } from '../../src/trajectory/TrajectoryStore.js'
import { MemoryStore } from '../../src/store/MemoryStore.js'
import { lookupSearch } from './fixtures/search.js'
import { type Experiment } from './helpers.js'

const SKIP = !process.env['VOLCENGINE_TOKEN'] || !process.env['VOLCENGINE_API_BASE']

// ─────────────────────────────── Tool ────────────────────────────────────────

const webSearchTool: ToolDefinition = {
  name:        'web_search',
  description: 'Search the web for information.',
  inputSchema: {
    type:       'object',
    properties: { query: { type: 'string' } },
    required:   ['query'],
  },
  parallelSafe: true,
  handler: async (input: unknown) => {
    const { query } = input as { query: string }
    return { query, result: lookupSearch(query) }
  },
}

// ─────────────────────────────── Skill Instructions ──────────────────────────

const SKILL_V1 = `## Research Guidelines (v1.0)
搜索相关信息，简洁地总结发现。专注于事实准确性。`

const SKILL_V2 = `## Research Guidelines (v1.1)
搜索相关信息，以带引用的格式总结发现。专注于事实准确性和来源可信度。
每个观点至少提供 2 个来源（格式：[来源 1]、[来源 2]）。`

// ─────────────────────────────── Agent Configs ───────────────────────────────

const MODEL = {
  provider: 'volcengine',
  model:    'doubao-seed-2.0-lite',
  adapter:  'openai-compatible' as const,
}

const skillTesterV1: AgentConfig = {
  agentId:      'skill-tester-v1',
  version:      '1.1.0',
  systemPrompt: `你是一个技术研究员，使用 web_search 查询技术信息并汇总报告。
执行顺序：
1. 首先调用 skill_request，请求加载 research skill。
2. skill_request 返回后，再调用 web_search 查询 TypeScript 5.0。
3. 根据加载后的 research skill 指令输出最终报告。`,
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 8 }] },
  model: MODEL,
  skills: { research: '1.0.0' },
  skillInstructions: { research: SKILL_V1 },
}

const skillTesterV2: AgentConfig = {
  agentId:      'skill-tester-v2',
  version:      '1.2.0',
  systemPrompt: `你是一个技术研究员，使用 web_search 查询技术信息并汇总报告。
执行顺序：
1. 首先调用 skill_request，请求加载 research skill。
2. skill_request 返回后，再调用 web_search 查询 TypeScript 5.0。
3. 根据加载后的 research skill 指令输出最终报告。`,
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 8 }] },
  model: MODEL,
  skills: { research: '1.1.0' },
  skillInstructions: { research: SKILL_V2 },
}

// ─────────────────────────────── Tests ───────────────────────────────────────

describe('Case 5: Skill A/B 版本对比', () => {
  let trajectoryStore: TrajectoryStore
  let r1: AgentResult
  let r2: AgentResult
  let t1: Trajectory
  let t2: Trajectory

  const goal = '分析 TypeScript 5.0 的主要新特性'

  beforeAll(async () => {
    if (SKIP) return

    trajectoryStore = new TrajectoryStore({ jsonlDir: './test-output/trajectories' })
    const milkie = new Milkie({
      stateStore:      new MemoryStore(),
      trajectoryStore,
      tools:           [webSearchTool],
    })

    milkie.registerAgent(skillTesterV1)
    milkie.registerAgent(skillTesterV2)

    // Run both agents in parallel (A/B test)
    ;[r1, r2] = await Promise.all([
      milkie.invoke({ agentId: 'skill-tester-v1', goal, input: goal }),
      milkie.invoke({ agentId: 'skill-tester-v2', goal, input: goal }),
    ])

    ;[t1, t2] = await Promise.all([
      trajectoryStore.getByRunId(r1.agentRunId),
      trajectoryStore.getByRunId(r2.agentRunId),
    ])
  }, 120_000)

  const live = SKIP ? it.skip : it

  live('两个 Agent 均完成', () => {
    expect(r1.status).toBe('completed')
    expect(r2.status).toBe('completed')
  })

  live('两个 trajectory 均有 llm.call span', () => {
    expect(t1.spans.filter(s => s.name === 'llm.call').length).toBeGreaterThan(0)
    expect(t2.spans.filter(s => s.name === 'llm.call').length).toBeGreaterThan(0)
  })

  live('两个 Agent 均调用了 web_search', () => {
    const search1 = t1.spans.filter(s => s.name === 'tool.call' && s.attributes['toolName'] === 'web_search')
    const search2 = t2.spans.filter(s => s.name === 'tool.call' && s.attributes['toolName'] === 'web_search')
    expect(search1.length).toBeGreaterThan(0)
    expect(search2.length).toBeGreaterThan(0)
  })

  live('两个 Agent 均通过 skill_request 渐进加载 research skill', () => {
    const skillRequests1 = t1.spans.filter(s => s.name === 'tool.call' && s.attributes['toolName'] === 'skill_request')
    const skillRequests2 = t2.spans.filter(s => s.name === 'tool.call' && s.attributes['toolName'] === 'skill_request')
    expect(skillRequests1.length).toBeGreaterThan(0)
    expect(skillRequests2.length).toBeGreaterThan(0)
    expect(skillRequests1[0]!.attributes['output']).toMatchObject({
      requested: 'research',
      status:    'pending_next_epoch',
      version:   '1.0.0',
    })
    expect(skillRequests2[0]!.attributes['output']).toMatchObject({
      requested: 'research',
      status:    'pending_next_epoch',
      version:   '1.1.0',
    })
  })

  live('research skill 在后续 llm.call 的 contextEpoch 中生效', () => {
    for (const trajectory of [t1, t2]) {
      const llmSpans = trajectory.spans.filter(s => s.name === 'llm.call')
      const firstWithSkill = llmSpans.find(s =>
        (s.attributes['loadedSkills'] as string[] | undefined)?.includes('research')
      )
      expect(firstWithSkill).toBeDefined()
      expect(firstWithSkill!.attributes['contextEpoch']).toBe(1)
      const firstLlm = llmSpans[0]!
      expect(firstLlm.attributes['loadedSkills']).toEqual([])
      expect(firstLlm.attributes['contextEpoch']).toBe(0)
    }
  })

  live('v1.2 的输出中包含 citation 相关内容（skill v1.1 新增要求的可观测效果）', () => {
    // v1.1 skill 要求 "每个观点至少提供 2 个来源"
    // 通过输出检查是否有引用格式
    expect(r2.output).toMatch(/来源|source|引用|reference|\[[^\]]+\]/i)
  })

  live('v1.1 和 v1.2 的输出不完全相同', () => {
    // The two versions should produce different outputs due to different skill instructions
    expect(r1.output).not.toBe(r2.output)
  })

  live('resolvedManifest 归因到 research skill 版本差异', () => {
    expect(t1.resolvedManifest?.agentVersion).toBe('1.1.0')
    expect(t2.resolvedManifest?.agentVersion).toBe('1.2.0')
    expect(t1.resolvedManifest?.skills['research']?.version).toBe('1.0.0')
    expect(t2.resolvedManifest?.skills['research']?.version).toBe('1.1.0')
    expect(t1.resolvedManifest?.model.model).toBe(t2.resolvedManifest?.model.model)
  })

  live('可构造 Experiment 对象用于后续分析', () => {
    const experiment: Experiment = {
      id:   'research-skill-upgrade',
      goal,
      variants: [
        { name: 'research-v1.0', agentVersion: '1.1.0', trajectoryIds: [t1.traceId] },
        { name: 'research-v1.1', agentVersion: '1.2.0', trajectoryIds: [t2.traceId] },
      ],
    }
    expect(experiment.variants).toHaveLength(2)
    expect(experiment.variants[0]!.trajectoryIds[0]).toBe(t1.traceId)
    expect(experiment.variants[1]!.trajectoryIds[0]).toBe(t2.traceId)
  })

  live('两个 trajectory 的 traceId 不同（独立执行）', () => {
    expect(t1.traceId).not.toBe(t2.traceId)
  })
})

// ─────────────────────────────── Skill Epoch Test ────────────────────────────

describe('Case 5: Instructions 动态加载（contextEpoch）', () => {
  const live = SKIP ? it.skip : it

  live('loadInstructions 递增 contextEpoch', () => {
    // Unit-level test for ContextLayer epoch mechanism
    const { ContextLayer } = require('../../src/context/ContextLayer.js')
    const ctx = new ContextLayer({ systemPrompt: 'test', model: 'test' })

    expect(ctx.getContextEpoch()).toBe(0)

    ctx.loadInstructions('research', SKILL_V1)
    expect(ctx.getContextEpoch()).toBe(1)

    ctx.loadInstructions('research', SKILL_V2)
    expect(ctx.getContextEpoch()).toBe(2)

    ctx.unloadInstructions('research')
    expect(ctx.getContextEpoch()).toBe(3)
  })

  live('已加载的 instructions 出现在 buildRequest system prompt 中', () => {
    const { ContextLayer } = require('../../src/context/ContextLayer.js')
    const { WorkingMemory } = require('../../src/store/WorkingMemory.js')
    const ctx = new ContextLayer({ systemPrompt: 'base prompt', model: 'test' })
    const mem = new WorkingMemory()

    ctx.loadInstructions('research', SKILL_V1)
    const request = ctx.buildRequest([], mem)
    expect(request.system).toContain('Research Guidelines')
    expect(request.system).toContain('v1.0')
  })
})
