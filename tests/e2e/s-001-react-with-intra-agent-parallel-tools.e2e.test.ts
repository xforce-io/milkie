/**
 * Case 1: Plan-and-Act — 竞品分析报告生成
 *
 * 验证：react FSM、Cognitive toolbox（create_plan / update_step）、intra-agent 并行
 * （单 Agent 单次响应多 tool_use）、write_file 写入真实文件、Trajectory
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
import { lookupSearch } from './fixtures/search.js'

const SKIP = !process.env['VOLCENGINE_TOKEN'] || !process.env['VOLCENGINE_API_BASE']

const OUTPUT_DIR = path.resolve('./test-output/case1')

// ─────────────────────────────── Tools ───────────────────────────────────────

const webSearchTool: ToolDefinition = {
  name:        'web_search',
  description: 'Search the web for information. Multiple calls can be made in parallel.',
  inputSchema: {
    type:       'object',
    properties: { query: { type: 'string', description: 'Search query' } },
    required:   ['query'],
  },
  parallelSafe: true,
  handler: async (input: unknown) => {
    const { query } = input as { query: string }
    return { query, result: lookupSearch(query) }
  },
}

const writeFileTool: ToolDefinition = {
  name:        'write_file',
  description: 'Write content to a file.',
  inputSchema: {
    type:       'object',
    properties: {
      path:    { type: 'string', description: 'File path to write to' },
      content: { type: 'string', description: 'Content to write' },
    },
    required: ['path', 'content'],
  },
  handler: async (input: unknown) => {
    const { path: filePath, content } = input as { path: string; content: string }
    const absPath = path.resolve(filePath)
    fs.mkdirSync(path.dirname(absPath), { recursive: true })
    fs.writeFileSync(absPath, content, 'utf-8')
    return { written: true, path: absPath }
  },
}

// ─────────────────────────────── Agent Config ────────────────────────────────

const analystConfig: AgentConfig = {
  agentId:      'analyst',
  version:      '1.0.0',
  systemPrompt: `你是一个竞品分析师，使用 cognitive toolbox 管理分析进度，用 web_search 和 write_file 完成报告。

步骤（严格按此顺序）：
1. 首先调用 create_plan，列出所有步骤：搜索产品A/B/C、对比分析、写报告。
2. 在同一次响应中同时调用 web_search 三次（分别搜索"Product A features pricing"、"Product B features pricing"、"Product C features pricing"）。必须在一次响应中同时调用全部三个。
3. 三次搜索全部返回后，调用 update_step 标记搜索步骤为 done，再调用 write_file 写入报告到 ./test-output/case1/report.md。
4. 报告写完后调用 update_step 将报告步骤标记为 done，然后输出完成信息。`,
  fsm: {
    states: [{ name: 'react', type: 'llm', max_iterations: 10 }],
  },
  model: {
    provider: 'volcengine',
    model:    'doubao-seed-2.0-lite',
    adapter:  'openai-compatible',
  },
}

// ─────────────────────────────── Test Setup ──────────────────────────────────

describe('Case 1: Plan-and-Act — 竞品分析', () => {
  let result: AgentResult
  let trajectory: Trajectory
  let milkie: Milkie
  let trajectoryStore: TrajectoryStore

  beforeAll(async () => {
    if (SKIP) return

    fs.mkdirSync(OUTPUT_DIR, { recursive: true })

    trajectoryStore = new TrajectoryStore({ jsonlDir: './test-output/trajectories' })
    milkie = new Milkie({
      stateStore:      new MemoryStore(),
      trajectoryStore,
      tools:           [webSearchTool, writeFileTool],
    })
    milkie.registerAgent(analystConfig)

    result = await milkie.invoke({
      agentId: 'analyst',
      goal:    '分析 Product A/B/C 的核心功能差异，生成竞品分析报告',
      input:   '输出 Markdown 报告到 ./test-output/case1/report.md',
    })
    trajectory = await trajectoryStore.getByRunId(result.agentRunId)
  }, 120_000)

  const live = SKIP ? it.skip : it

  live('Agent 执行完成', () => {
    expect(result.status).toBe('completed')
  })

  live('报告文件存在且包含三个产品内容', () => {
    const reportPath = path.join(OUTPUT_DIR, 'report.md')
    expect(fs.existsSync(reportPath)).toBe(true)
    const report = fs.readFileSync(reportPath, 'utf-8')
    expect(report).toMatch(/Product A/i)
    expect(report).toMatch(/Product B/i)
    expect(report).toMatch(/Product C/i)
    expect(report.length).toBeGreaterThan(100)
  })

  live('create_plan 在 write_file 之前调用（先规划后输出）', () => {
    const spans = trajectory.spans.filter(s => s.name === 'tool.call')
    const planIdx  = spans.findIndex(s => s.attributes['toolName'] === 'create_plan')
    const writeIdx = spans.findIndex(s => s.attributes['toolName'] === 'write_file')
    expect(planIdx).toBeGreaterThanOrEqual(0)
    expect(writeIdx).toBeGreaterThanOrEqual(0)
    // create_plan must be called before the final report is written
    expect(planIdx).toBeLessThan(writeIdx)
  })

  live('create_plan 写入 workingMemory，返回步骤列表', () => {
    const planSpan = trajectory.spans.find(
      s => s.name === 'tool.call' && s.attributes['toolName'] === 'create_plan'
    )
    expect(planSpan).toBeDefined()
    const planOutput = planSpan!.attributes['output'] as { steps: unknown[] } | undefined
    expect(planOutput?.steps).toBeDefined()
    expect(planOutput!.steps.length).toBeGreaterThanOrEqual(2)
  })

  live('web_search 被调用 3 次', () => {
    const searchSpans = trajectory.spans.filter(
      s => s.name === 'tool.call' && s.attributes['toolName'] === 'web_search'
    )
    expect(searchSpans).toHaveLength(3)
  })

  live('3 个 web_search 在同一 turn 内（intra-agent 并行）', () => {
    const searchSpans = trajectory.spans.filter(
      s => s.name === 'tool.call' && s.attributes['toolName'] === 'web_search'
    )
    expect(searchSpans).toHaveLength(3)
    const turns = searchSpans.map(s => s.attributes['turn'] as number)
    // All 3 searches should be in the same LLM turn (one response with 3 tool_use blocks)
    expect(new Set(turns).size).toBe(1)
  })

  live('update_step 将步骤标记为 done（checklist 模式）', () => {
    const updateSpans = trajectory.spans.filter(
      s => s.name === 'tool.call' && s.attributes['toolName'] === 'update_step'
    )
    expect(updateSpans.length).toBeGreaterThanOrEqual(1)
    const doneUpdates = updateSpans.filter(
      s => (s.attributes['input'] as { status?: string })?.status === 'done'
    )
    expect(doneUpdates.length).toBeGreaterThanOrEqual(1)
  })

  live('llm.call span 记录 provider 和 model', () => {
    const llmSpans = trajectory.spans.filter(s => s.name === 'llm.call')
    expect(llmSpans.length).toBeGreaterThan(0)
    for (const span of llmSpans) {
      expect(span.attributes['provider']).toBe('volcengine')
      expect(span.attributes['model']).toContain('doubao')
    }
  })

  live('无 agent.spawn span（单 Agent，无 sub-agent）', () => {
    const spawnSpans = trajectory.spans.filter(s => s.name === 'agent.spawn')
    expect(spawnSpans).toHaveLength(0)
  })

  live('Trajectory 以 JSONL 文件落盘', () => {
    const file = path.resolve('./test-output/trajectories', `analyst-${trajectory.traceId}.jsonl`)
    expect(fs.existsSync(file)).toBe(true)
    const lines = fs.readFileSync(file, 'utf-8').trim().split('\n')
    expect(lines.length).toBeGreaterThan(0)
    expect(JSON.parse(lines[0]!).span.name).toBeDefined()
  })
})
