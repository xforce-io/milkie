/**
 * Issue #166 — Milkie HER adapter (S-011 Path D)
 *
 * Two layers of coverage:
 *   1. Handler-level behavior of `makeEntityResolverTools` against a stub
 *      `ToolContext` — precise checks for utterance/pinned isolation, WM writes,
 *      emit, and every CommitOutput status branch.
 *   2. A full FSM e2e: a deterministic stub gateway drives `lookup_entity` →
 *      `commit_entity` across all four hierarchy levels inside a `collecting_slots`
 *      state, proving the adapter runs in-process and SLOTS_COMPLETE advances the
 *      FSM to `confirming`. No subprocess, no Redis, no live model.
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import type { AgentConfig } from '../../../../src/types/agent.js'
import type { AgentResult, Message } from '../../../../src/types/common.js'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../../../src/types/model.js'
import type { ToolContext } from '../../../../src/types/tool.js'
import type { Trajectory } from '../../../../src/types/trajectory.js'
import { Milkie } from '../../../../src/runtime/Milkie.js'
import { MemoryStore } from '../../../../src/store/MemoryStore.js'
import { WorkingMemory } from '../../../../src/store/WorkingMemory.js'
import { MemoryEventStore } from '../../../../src/trace/MemoryEventStore.js'
import { checkpointFromEvents } from '../../../../src/trace/diagnostics/checkpointFromEvents.js'
import { TrajectoryStore } from '../../../../src/trajectory/TrajectoryStore.js'

import { EntityResolver, type Schema } from '../../resolver/EntityResolver.js'
import { makeEntityResolverTools } from '../tools/entity-resolver.js'

// ─────────────────────────────── Fixtures ────────────────────────────────────

const schema = JSON.parse(
  readFileSync(join(__dirname, '../../resolver/schema.json'), 'utf8'),
) as Schema
const csv = readFileSync(join(__dirname, '../../resolver/data.csv'), 'utf8')

const REQUIRED = ['site', 'building', 'department', 'assignee']

// resolver is constructed ONCE — handlers must never re-parse.
const resolver = EntityResolver.load(schema, csv)
const tools = makeEntityResolverTools(resolver, REQUIRED)
const lookupTool = tools.find(t => t.name === 'lookup_entity')!
const commitTool = tools.find(t => t.name === 'commit_entity')!

function makeCtx(
  currentTurn: string,
  preset: Record<string, string> = {},
): { ctx: ToolContext; emitted: string[]; wm: WorkingMemory } {
  const emitted: string[] = []
  const wm = new WorkingMemory()
  for (const [k, v] of Object.entries(preset)) wm.set(k, v)
  const ctx = {
    workingMemory: wm,
    emit: (event: string) => { emitted.push(event) },
    currentTurn,
  } as unknown as ToolContext
  return { ctx, emitted, wm }
}

// ─────────────────────────── Adapter shape (AC2, 11, 12) ──────────────────────

describe('makeEntityResolverTools — shape & isolation', () => {
  it('exposes exactly two tools: lookup_entity and commit_entity (AC2)', () => {
    expect(tools).toHaveLength(2)
    expect(tools.map(t => t.name).sort()).toEqual(['commit_entity', 'lookup_entity'])
  })

  it('lookup_entity input schema exposes no utterance/query/pinned field (AC11, AC12)', () => {
    const schemaStr = JSON.stringify(lookupTool.inputSchema)
    expect(schemaStr).not.toMatch(/utterance/i)
    expect(schemaStr).not.toMatch(/query/i)
    expect(schemaStr).not.toMatch(/pinned/i)
  })

  it('commit_entity input schema exposes no utterance/query/pinned field (AC12)', () => {
    const schemaStr = JSON.stringify(commitTool.inputSchema)
    expect(schemaStr).not.toMatch(/utterance/i)
    expect(schemaStr).not.toMatch(/query/i)
    expect(schemaStr).not.toMatch(/pinned/i)
  })
})

// ─────────────────────────────── lookup_entity ───────────────────────────────

describe('lookup_entity handler', () => {
  it('reads utterance from ctx.currentTurn, not from tool input (AC11)', async () => {
    const { ctx } = makeCtx('总部')
    const out = (await lookupTool.handler({ context: { level: 'site' } }, ctx)) as {
      candidates: unknown[]; options: string[]; suggested: string | null
    }
    expect(out.options).toContain('S01')
    expect(out.suggested).toBe('S01')
  })

  it('returns { candidates, options, suggested } to the LLM (AC5)', async () => {
    const { ctx } = makeCtx('总部')
    const out = (await lookupTool.handler({ context: { level: 'site' } }, ctx)) as Record<string, unknown>
    expect(out).toHaveProperty('candidates')
    expect(out).toHaveProperty('options')
    expect(out).toHaveProperty('suggested')
  })

  it('derives pinned ancestors from WM so a child lookup is branch-filtered (AC12)', async () => {
    // site pinned to S01 (总部) → 主楼 lookup must not surface B03 (分楼, under S02)
    const { ctx } = makeCtx('主楼', { site: 'S01' })
    const out = (await lookupTool.handler({ context: { level: 'building' } }, ctx)) as {
      options: string[]; suggested: string | null
    }
    expect(out.options).toEqual(['B01'])
    expect(out.suggested).toBe('B01')
  })

  it('never writes WM nor emits (AC: lookup is read-only)', async () => {
    const { ctx, emitted, wm } = makeCtx('总部')
    await lookupTool.handler({ context: { level: 'site' } }, ctx)
    expect(emitted).toEqual([])
    expect(wm.get('site')).toBeUndefined()
  })
})

// ─────────────────────────────── commit_entity ───────────────────────────────

describe('commit_entity handler', () => {
  it('commit of a previously-suggested option writes WM = resolved.id (AC6, AC7)', async () => {
    const { ctx, emitted, wm } = makeCtx('总部')
    const lookup = (await lookupTool.handler({ context: { level: 'site' } }, ctx)) as {
      suggested: string | null
    }
    const selected = lookup.suggested!
    const out = (await commitTool.handler(
      { selected, context: { level: 'site' } },
      ctx,
    )) as { resolved: { id: string } }
    expect(out.resolved.id).toBe(selected)   // resolved.id came from prior lookup
    expect(wm.get('site')).toBe('S01')
    expect(emitted).toEqual([])              // not all slots yet
  })

  it('emits SLOTS_COMPLETE once every required level is in WM (AC8)', async () => {
    const { ctx, emitted, wm } = makeCtx('王芳', { site: 'S01', building: 'B01', department: 'D03' })
    const out = (await commitTool.handler(
      { selected: 'E008', context: { level: 'assignee' } },
      ctx,
    )) as { resolved: { id: string } }
    expect(out.resolved.id).toBe('E008')
    expect(wm.get('assignee')).toBe('E008')
    expect(emitted).toEqual(['SLOTS_COMPLETE'])
  })

  it('unknown selection → validationError, no WM write, no emit (AC9)', async () => {
    const { ctx, emitted, wm } = makeCtx('总部')
    const out = (await commitTool.handler(
      { selected: 'ZZZ', context: { level: 'site' } },
      ctx,
    )) as { validationError?: string }
    expect(out.validationError).toBeDefined()
    expect(wm.get('site')).toBeUndefined()
    expect(emitted).toEqual([])
  })

  it('wrong-level selection → missing → validationError, no WM write (AC9)', async () => {
    const { ctx, emitted, wm } = makeCtx('总部')
    // B01 is a building id; committing it at the site level is an under-selection.
    const out = (await commitTool.handler(
      { selected: 'B01', context: { level: 'site' } },
      ctx,
    )) as { validationError?: string }
    expect(out.validationError).toBeDefined()
    expect(wm.get('site')).toBeUndefined()
    expect(emitted).toEqual([])
  })

  it('corrected path writes WM = corrected id and returns corrected levels (AC10)', async () => {
    // pinned site/building = S01/B01; selecting E012 (张伟 under B02/D07) conflicts
    // with the pinned branch → auto-corrected to the same-label sibling E007 (B01/D03).
    const { ctx, emitted, wm } = makeCtx('张伟', { site: 'S01', building: 'B01' })
    const out = (await commitTool.handler(
      { selected: 'E012', context: { level: 'assignee' } },
      ctx,
    )) as { resolved: { id: string }; corrected: Record<string, string> }
    expect(out.resolved.id).toBe('E007')          // pinned ancestor wins
    expect(wm.get('assignee')).toBe('E007')        // WM holds the corrected id
    // `corrected` is a Record<string,string> (level → corrected id), NOT a plain
    // string — it can name a level other than the committed one. Lock that shape
    // so consumers cannot regress to a string contract.
    expect(typeof out.corrected).toBe('object')
    expect(typeof out.corrected.department).toBe('string')
    expect(out.corrected).toMatchObject({ department: 'D03' })
    expect(emitted).toEqual([])                     // department slot absent → no emit
  })
})

// ─────────────────────────── FSM e2e (AC1, AC4, AC8, AC11) ────────────────────

/**
 * Deterministic gateway: per turn it issues `lookup_entity` for the next unfilled
 * level (no utterance in the call — the adapter reads it from ctx.currentTurn),
 * then `commit_entity` selecting the value the lookup suggested. Level progression
 * is tracked by counting prior successful commits; the LLM never invents an id.
 */
class HerSlotGateway implements IModelGateway {
  private readonly levelOrder = ['site', 'building', 'department', 'assignee']
  private committed = 0

  async complete(request: ModelRequest): Promise<ModelResponse> {
    const messages = request.messages
    const last = messages[messages.length - 1]

    // All four levels resolved → confirming (terminal) state: final text.
    if (this.committed >= this.levelOrder.length) {
      return this.text('已为您登记报修负责人，请确认。')
    }
    const level = this.levelOrder[this.committed]!

    if (last && last.role === 'tool') {
      const producedBy = this.precedingToolUse(messages)
      const result = this.parseToolResult(last)
      if (producedBy === 'lookup_entity') {
        const opts = (result?.['options'] as string[] | undefined) ?? []
        const selected = (result?.['suggested'] as string | null) ?? opts[0]
        return this.tool(`commit-${level}`, 'commit_entity', { selected, context: { level } })
      }
      if (producedBy === 'commit_entity') {
        // This level is committed; close the turn and wait for the next utterance.
        this.committed++
        return this.text('已记录。')
      }
    }

    // Fresh user turn → look the next level up.
    return this.tool(`lookup-${level}`, 'lookup_entity', { context: { level } })
  }

  async *stream(_request: ModelRequest): AsyncIterable<never> {
    yield* []
  }

  private precedingToolUse(messages: Message[]): string | undefined {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i]!
      if (m.role === 'assistant') {
        const tu = m.content.find(c => c.type === 'tool_use')
        if (tu && tu.type === 'tool_use') return tu.name
      }
    }
    return undefined
  }

  private parseToolResult(m: Message): Record<string, unknown> | null {
    const part = m.content.find(c => c.type === 'tool_result')
    if (!part || part.type !== 'tool_result') return null
    try {
      return JSON.parse(part.content) as Record<string, unknown>
    } catch {
      return null
    }
  }

  private text(text: string): ModelResponse {
    return { content: [{ type: 'text', text }], toolCalls: [], finishReason: 'end_turn' }
  }

  private tool(id: string, name: string, input: unknown): ModelResponse {
    return {
      content:      [{ type: 'tool_use', id, name, input }],
      toolCalls:    [{ id, name, input }],
      finishReason: 'tool_use',
    }
  }
}

const repairAgentConfig: AgentConfig = {
  agentId:      'repair-ticketing',
  version:      '1.0.0',
  systemPrompt: '你负责为报修工单定位层级实体（站点 → 楼宇 → 部门 → 负责人）。',
  fsm: {
    states: [
      {
        name:  'collecting_slots',
        type:  'llm',
        tools: ['lookup_entity', 'commit_entity'],
        on:    { SLOTS_COMPLETE: 'confirming' },
      },
      {
        // Path D ends collecting_slots → confirming, where the agent waits for the
        // user to confirm before executing (… → executing → completed). It is a
        // wait-for-user state, NOT terminal: a terminal turn does not persist a
        // checkpoint, which would drop the final slot write (assignee) made on the
        // turn that fired SLOTS_COMPLETE.
        name:         'confirming',
        type:         'llm',
        tools:        [],
        instructions: '向用户确认已登记的报修负责人。',
      },
    ],
  },
  model: { provider: 'volcengine', model: 'doubao-seed-2.0-lite', adapter: 'openai-compatible' },
}

async function readCheckpoint(
  stateStore: MemoryStore,
  eventStore: MemoryEventStore,
  contextId: string,
): Promise<{ context: { workingMemory: { data: Record<string, unknown> } }; fsm: { currentState: string } } | null> {
  const runId = (await stateStore.get(`context:${contextId}:checkpoint-run:latest`)) as string | undefined
  if (!runId) return null
  return checkpointFromEvents(await eventStore.readByRunId(runId)) as never
}

describe('Path D: hierarchical slot filling inside collecting_slots (FSM e2e)', () => {
  let stateStore: MemoryStore
  let eventStore: MemoryEventStore
  let trajectoryStore: TrajectoryStore
  let milkie: Milkie
  let trajectory: Trajectory
  let lastResult: AgentResult

  const contextId = `ctx-repair-pathD-${Date.now()}`
  const goal = '为报修工单定位负责人'

  async function sendTurn(input: string): Promise<AgentResult> {
    return milkie.invoke({ agentId: 'repair-ticketing', goal, input, contextId })
  }

  beforeAll(async () => {
    stateStore = new MemoryStore()
    eventStore = new MemoryEventStore()
    trajectoryStore = new TrajectoryStore({ jsonlDir: './test-output/trajectories' })
    milkie = new Milkie({
      stateStore,
      eventStore,
      trajectoryStore,
      gateway: new HerSlotGateway(),
      tools,
    })
    milkie.registerAgent(repairAgentConfig)

    await sendTurn('总部')         // → site S01
    await sendTurn('主楼')         // → building B01
    await sendTurn('IT网络部')     // → department D03
    lastResult = await sendTurn('王芳')  // → assignee E008 → SLOTS_COMPLETE

    trajectory = await trajectoryStore.getByContextId(contextId)
  }, 60_000)

  it('FSM state names never equal a hierarchy-level name (AC4)', () => {
    const stateNames = repairAgentConfig.fsm.states.map(s => s.name)
    expect(stateNames.filter(n => REQUIRED.includes(n))).toEqual([])
  })

  it('lookup_entity tool calls carry no utterance/query field in their input (AC11)', () => {
    const lookupSpans = trajectory.spans.filter(
      s => s.name === 'tool.call' && s.attributes['toolName'] === 'lookup_entity',
    )
    expect(lookupSpans.length).toBeGreaterThanOrEqual(4)
    for (const span of lookupSpans) {
      const input = span.attributes['input'] as Record<string, unknown>
      expect(input).not.toHaveProperty('utterance')
      expect(input).not.toHaveProperty('query')
      expect((input['context'] as { level?: string }).level).toBeDefined()
    }
  })

  it('every level resolves in-process and WM holds the four resolved ids (AC1, AC7)', async () => {
    const cp = await readCheckpoint(stateStore, eventStore, contextId)
    const wm = cp?.context.workingMemory.data ?? {}
    expect(wm).toMatchObject({
      site:       'S01',
      building:   'B01',
      department: 'D03',
      assignee:   'E008',
    })
  })

  it('SLOTS_COMPLETE advances the FSM to confirming (AC8)', () => {
    const states = trajectory.spans
      .filter(s => s.name === 'fsm.transition')
      .map(s => s.attributes['toState'] as string)
    expect(states).toContain('confirming')
    expect(lastResult.status).toBe('completed')
  })
})
