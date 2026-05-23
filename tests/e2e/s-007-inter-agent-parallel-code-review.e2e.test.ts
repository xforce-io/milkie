/**
 * Case 2: Inter-Agent 并行 — 多角色代码审查
 *
 * 验证：Named sub-agent tools、inter-agent 并行（allSettled join）、
 * Context 隔离（各 sub-agent 独立 FSM + Context）、agent.spawn span
 */

import fs from 'fs'
import path from 'path'
import type { AgentConfig } from '../../src/types/agent.js'
import type { AgentResult } from '../../src/types/common.js'
import type { Trajectory } from '../../src/types/trajectory.js'
import type { ToolDefinition } from '../../src/types/tool.js'
import { Milkie } from '../../src/runtime/Milkie.js'
import { TrajectoryStore } from '../../src/trajectory/TrajectoryStore.js'
import { MemoryStore } from '../../src/store/MemoryStore.js'

const SKIP = !process.env['VOLCENGINE_TOKEN'] || !process.env['VOLCENGINE_API_BASE']

// ─────────────────────────────── Tools ───────────────────────────────────────

const readFileTool: ToolDefinition = {
  name:        'read_file',
  description: 'Read the contents of a file.',
  inputSchema: {
    type:       'object',
    properties: { path: { type: 'string', description: 'File path to read' } },
    required:   ['path'],
  },
  handler: async (input: unknown) => {
    const { path: filePath } = input as { path: string }
    let absPath = path.resolve(filePath)
    if (!fs.existsSync(absPath) && path.basename(filePath) === 'target.ts') {
      absPath = path.resolve(FIXTURE_PATH)
    }
    if (!fs.existsSync(absPath)) return { error: `File not found: ${absPath}` }
    return { path: absPath, content: fs.readFileSync(absPath, 'utf-8') }
  },
}

// ─────────────────────────────── Agent Configs ───────────────────────────────

const MODEL = {
  provider: 'volcengine',
  model:    'doubao-seed-2.0-lite',
  adapter:  'openai-compatible' as const,
}

const FIXTURE_PATH = './tests/e2e/fixtures/code/target.ts'

const securityReviewerConfig: AgentConfig = {
  agentId:      'security-reviewer',
  version:      '1.0.0',
  systemPrompt: '你是一位安全审查专家。只调用一次 read_file 读取用户给出的代码文件路径，然后必须输出简洁的安全审查报告。重点检查 SQL 注入、XSS、身份验证问题。报告必须包含“SQL 注入”或“安全”关键词。',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 8 }] },
  model: MODEL,
}

const perfReviewerConfig: AgentConfig = {
  agentId:      'perf-reviewer',
  version:      '1.0.0',
  systemPrompt: '你是一位性能审查专家。只调用一次 read_file 读取用户给出的代码文件路径，然后必须输出简洁的性能审查报告。重点检查 N+1 查询、不必要的循环、内存泄漏。报告必须包含“N+1”或“性能”关键词。',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 8 }] },
  model: MODEL,
}

const styleCheckerConfig: AgentConfig = {
  agentId:      'style-checker',
  version:      '1.0.0',
  systemPrompt: '你是一位代码风格审查专家。只调用一次 read_file 读取用户给出的代码文件路径，然后必须输出简洁的风格审查报告。重点检查命名规范、代码格式、注释质量。报告必须包含“命名”或“风格”关键词。',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 8 }] },
  model: MODEL,
}

const orchestratorConfig: AgentConfig = {
  agentId:      'review-orchestrator',
  version:      '1.0.0',
  systemPrompt: `你是代码审查协调员。对提交的代码同时启动三个专项审查。

重要：在一次响应中同时调用 security-reviewer、perf-reviewer、style-checker 三个工具，不要分开调用。
每个工具都传入 goal 和 input 参数，input 必须包含原始代码文件路径。
等收到全部三个结果后，汇总为最终审查报告。最终报告必须分别包含安全、性能、风格三段。`,
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 10 }] },
  model: MODEL,
  subAgents: {
    'security-reviewer': '1.0.0',
    'perf-reviewer':     '1.0.0',
    'style-checker':     '1.0.0',
  },
}

// ─────────────────────────────── Test Setup ──────────────────────────────────

describe('Case 2: Inter-Agent 并行 — 多角色代码审查', () => {
  let result: AgentResult
  let trajectory: Trajectory
  let milkie: Milkie
  let trajectoryStore: TrajectoryStore

  beforeAll(async () => {
    if (SKIP) return

    trajectoryStore = new TrajectoryStore({ jsonlDir: './test-output/trajectories' })
    milkie = new Milkie({
      stateStore:      new MemoryStore(),
      trajectoryStore,
      tools:           [readFileTool],
    })

    milkie.registerAgent(securityReviewerConfig)
    milkie.registerAgent(perfReviewerConfig)
    milkie.registerAgent(styleCheckerConfig)
    milkie.registerAgent(orchestratorConfig)

    result = await milkie.invoke({
      agentId: 'review-orchestrator',
      goal:    `审查代码文件 ${FIXTURE_PATH}，发现安全、性能和风格问题`,
      input:   `请并行启动三个专项审查。代码文件路径：${FIXTURE_PATH}`,
    })

    trajectory = await trajectoryStore.getByRunId(result.agentRunId)
  }, 180_000)

  const live = SKIP ? it.skip : it

  live('Orchestrator 执行完成', () => {
    expect(result.status).toBe('completed')
  })

  live('生成至少 3 个 agent.spawn span（inter-agent 并行）', () => {
    const spawnSpans = trajectory.spans.filter(s => s.name === 'agent.spawn')
    expect(spawnSpans.length).toBeGreaterThanOrEqual(3)
  })

  live('3 个子 Agent 均被调用', () => {
    const spawnSpans = trajectory.spans.filter(s => s.name === 'agent.spawn')
    const childAgents = spawnSpans.map(s => s.attributes['childAgentId'] as string)
    expect(childAgents).toContain('security-reviewer')
    expect(childAgents).toContain('perf-reviewer')
    expect(childAgents).toContain('style-checker')
  })

  live('3 个 sub-agent 的 childTraceId 各自独立（Context 隔离）', () => {
    const spawnSpans = trajectory.spans.filter(s => s.name === 'agent.spawn')
    const childTraceIds = spawnSpans.map(s => s.attributes['childTraceId'] as string)
    // All 3 should have distinct traceIds
    expect(new Set(childTraceIds).size).toBe(childTraceIds.length)
    for (const id of childTraceIds) {
      expect(id).toBeTruthy()
    }
  })

  live('3 个 sub-agent 拥有独立 contextId', () => {
    const childRunSpans = trajectory.spans.filter(
      s => s.name === 'agent.run' && s.attributes['agentId'] !== 'review-orchestrator'
    )
    const childContextIds = childRunSpans.map(s => s.attributes['contextId'] as string)
    expect(childContextIds.length).toBeGreaterThanOrEqual(3)
    expect(new Set(childContextIds).size).toBe(childContextIds.length)
  })

  live('3 个 sub-agent spans 使用真实独立 traceId', () => {
    const spawnSpans = trajectory.spans.filter(s => s.name === 'agent.spawn')
    const childTraceIds = new Set(spawnSpans.map(s => s.attributes['childTraceId'] as string))
    const childRunSpans = trajectory.spans.filter(
      s => s.name === 'agent.run' && s.attributes['agentId'] !== 'review-orchestrator'
    )
    const actualChildTraceIds = childRunSpans.map(s => s.traceId)
    expect(actualChildTraceIds).toHaveLength(3)
    expect(new Set(actualChildTraceIds)).toEqual(childTraceIds)
    expect(actualChildTraceIds.every(id => id !== trajectory.traceId)).toBe(true)
  })

  live('3 个 sub-agent 在同一 turn 内并发启动', () => {
    const spawnSpans = trajectory.spans.filter(s => s.name === 'agent.spawn')
    const turns = spawnSpans.map(s => s.attributes['turn'] as number)
    expect(new Set(turns).size).toBe(1)
  })

  live('所有 TaskResult 均为 success', () => {
    const spawnSpans = trajectory.spans.filter(s => s.name === 'agent.spawn')
    for (const span of spawnSpans) {
      expect(span.attributes['resultStatus']).toBe('completed')
    }
  })

  live('ResolvedManifest 记录三个 sub-agent 版本', () => {
    expect(trajectory.resolvedManifest?.subAgents['security-reviewer']?.version).toBe('1.0.0')
    expect(trajectory.resolvedManifest?.subAgents['perf-reviewer']?.version).toBe('1.0.0')
    expect(trajectory.resolvedManifest?.subAgents['style-checker']?.version).toBe('1.0.0')
  })

  live('Orchestrator output 包含三类审查发现', () => {
    expect(result.output).toMatch(/SQL|注入|安全|injection/i)
    expect(result.output).toMatch(/N\+1|性能|性能问题|查询/i)
    expect(result.output).toMatch(/命名|风格|规范|naming/i)
  })
})
