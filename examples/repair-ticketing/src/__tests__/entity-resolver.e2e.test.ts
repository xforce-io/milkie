/**
 * Issue #166 / #175 — Milkie HER adapter (S-011 Path D), single-state tier.
 *
 * Two layers of coverage:
 *   1. Handler-level behavior of `makeEntityResolverTools` against a stub
 *      `ToolContext` — precise checks for utterance/pinned isolation, WM writes,
 *      and every CommitOutput status branch.
 *   2. A full e2e: a deterministic stub gateway drives `lookup_entity` →
 *      `commit_entity` across all four hierarchy levels inside ONE autonomous
 *      `llm` state, proving the adapter runs in-process and the four resolved ids
 *      land in WM. No subprocess, no Redis, no live model.
 *
 * #175 de-core: there is no business event and no multi-state FSM anymore. The
 * adapter no longer calls `ctx.emit('SLOTS_COMPLETE')`; "all slots filled" is a
 * plain fact in working memory (read back by the assemble_ticket precondition in
 * the full agent — see repair-ticketing.e2e.test.ts). So this file asserts WM
 * state and tool-call shape, not emitted events or `fsm.transition` spans.
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
): { ctx: ToolContext; wm: WorkingMemory } {
  const wm = new WorkingMemory()
  for (const [k, v] of Object.entries(preset)) wm.set(k, v)
  const ctx = {
    workingMemory: wm,
    currentTurn,
  } as unknown as ToolContext
  return { ctx, wm }
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

  it('never writes WM (AC: lookup is read-only)', async () => {
    const { ctx, wm } = makeCtx('总部')
    await lookupTool.handler({ context: { level: 'site' } }, ctx)
    expect(wm.get('site')).toBeUndefined()
  })
})

// ─────────────────────────────── commit_entity ───────────────────────────────

describe('commit_entity handler', () => {
  it('commit of a previously-suggested option writes WM = resolved.id (AC6, AC7)', async () => {
    const { ctx, wm } = makeCtx('总部')
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
  })

  it('tolerates a stringified `context` and keys WM by the level name, not a numeric fallback', async () => {
    // Regression (#178): some models (e.g. DeepSeek) serialize the nested `context`
    // object as a JSON *string*. Unparsed, `context.level` reads undefined and the
    // slot lands under a numeric path-length key (e.g. "1"), so requiredSlots stay
    // permanently incomplete. The handler must parse the string form.
    const { ctx, wm } = makeCtx('总部')
    const out = (await commitTool.handler(
      { selected: 'S01', context: '{"level":"site"}' },   // context as a JSON string
      ctx,
    )) as { resolved?: { id: string }; validationError?: string }
    expect(out.validationError).toBeUndefined()
    expect(out.resolved?.id).toBe('S01')
    expect(wm.get('site')).toBe('S01')      // keyed by level name…
    expect(wm.get('1')).toBeUndefined()     // …NOT the numeric path-length fallback
  })

  it('rejects a commit with no resolvable level instead of writing a numeric key', async () => {
    const { ctx, wm } = makeCtx('总部')
    const out = (await commitTool.handler(
      { selected: 'S01' },                  // no context at all → no level
      ctx,
    )) as { validationError?: string }
    expect(out.validationError).toBeDefined()
    expect(wm.get('site')).toBeUndefined()
    expect(wm.get('1')).toBeUndefined()
  })

  it('writes the last required level into WM, completing the slot set (AC8)', async () => {
    // #175: with three levels pinned, committing the fourth simply lands the id in
    // WM — there is no business event (ctx.emit removed). "All slots filled" is
    // read back from WM by the assemble_ticket precondition.
    const { ctx, wm } = makeCtx('王芳', { site: 'S01', building: 'B01', department: 'D03' })
    const out = (await commitTool.handler(
      { selected: 'E008', context: { level: 'assignee' } },
      ctx,
    )) as { resolved: { id: string } }
    expect(out.resolved.id).toBe('E008')
    expect(wm.get('assignee')).toBe('E008')
    // every required level is now present in WM
    expect(REQUIRED.every(level => wm.get(level) != null && wm.get(level) !== '')).toBe(true)
  })

  it('unknown selection → validationError, no WM write (AC9)', async () => {
    const { ctx, wm } = makeCtx('总部')
    const out = (await commitTool.handler(
      { selected: 'ZZZ', context: { level: 'site' } },
      ctx,
    )) as { validationError?: string }
    expect(out.validationError).toBeDefined()
    expect(wm.get('site')).toBeUndefined()
  })

  it('wrong-level selection → missing → validationError, no WM write (AC9)', async () => {
    const { ctx, wm } = makeCtx('总部')
    // B01 is a building id; committing it at the site level is an under-selection.
    const out = (await commitTool.handler(
      { selected: 'B01', context: { level: 'site' } },
      ctx,
    )) as { validationError?: string }
    expect(out.validationError).toBeDefined()
    expect(wm.get('site')).toBeUndefined()
  })

  it('corrected path writes WM = corrected id and returns corrected levels (AC10)', async () => {
    // pinned site/building = S01/B01; selecting E012 (张伟 under B02/D07) conflicts
    // with the pinned branch → auto-corrected to the same-label sibling E007 (B01/D03).
    const { ctx, wm } = makeCtx('张伟', { site: 'S01', building: 'B01' })
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
  })
})

// ───────────────────── Single-state e2e (AC1, AC4, AC7, AC11) ─────────────────

/**
 * Deterministic gateway driving the four levels inside ONE autonomous `llm` state
 * and ONE turn (#175 — the de-cored runtime no longer steps user-state →
 * user-state, and no longer persists WM across completed turns). It issues
 * `lookup_entity` for the next unfilled level (no utterance in the call — the
 * adapter reads it from ctx.currentTurn), then `commit_entity` selecting the value
 * the lookup suggested; the `pinned`-filtered resolver disambiguates each level
 * out of the one full utterance. After all four commit it ends the turn with
 * text. The LLM never invents an id.
 */
class HerSlotGateway implements IModelGateway {
  private readonly levelOrder = ['site', 'building', 'department', 'assignee']
  private committed = 0

  async complete(request: ModelRequest): Promise<ModelResponse> {
    const messages = request.messages
    const last = messages[messages.length - 1]

    if (last && last.role === 'tool') {
      const producedBy = this.precedingToolUse(messages)
      const result = this.parseToolResult(last)
      if (producedBy === 'lookup_entity') {
        const level = this.levelOrder[this.committed]!
        const opts = (result?.['options'] as string[] | undefined) ?? []
        const selected = (result?.['suggested'] as string | null) ?? opts[0]
        return this.tool(`commit-${level}`, 'commit_entity', { selected, context: { level } })
      }
      if (producedBy === 'commit_entity') {
        this.committed++
        // Stay in the same turn: drive the next level, or finish once all four
        // are committed.
        if (this.committed < this.levelOrder.length) {
          const level = this.levelOrder[this.committed]!
          return this.tool(`lookup-${level}`, 'lookup_entity', { context: { level } })
        }
        return this.text('已为您登记报修负责人，请确认。')
      }
    }

    // Fresh user turn → look the first level up.
    return this.tool('lookup-site', 'lookup_entity', { context: { level: 'site' } })
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
  version:      '2.0.0',
  systemPrompt: '你负责为报修工单定位层级实体（站点 → 楼宇 → 部门 → 负责人）。',
  // #175: a single autonomous llm state. No `on:` edges, no business event — slot
  // completeness is just a fact in WM.
  fsm: {
    states: [
      {
        name:  'collecting_slots',
        type:  'llm',
        tools: ['lookup_entity', 'commit_entity'],
      },
    ],
  },
  model: { provider: 'volcengine', model: 'doubao-seed-2.0-lite', adapter: 'openai-compatible' },
}

/** The cumulative WM field map for a run = its last `wm.mutated` snapshot (each
 *  snapshot is the full WorkingMemory, not a delta). #175: read from the event log
 *  directly — completed turns no longer persist a checkpoint. */
async function latestWorkingMemory(
  eventStore: MemoryEventStore,
  runId: string,
): Promise<Record<string, unknown>> {
  let data: Record<string, unknown> = {}
  for (const e of await eventStore.readByRunId(runId)) {
    if (e.type === 'wm.mutated') {
      const snap = (e.payload as { snapshot?: { data?: Record<string, unknown> } }).snapshot
      data = snap?.data ?? {}
    }
  }
  return data
}

describe('Path D: hierarchical slot filling inside a single autonomous state (e2e)', () => {
  let eventStore: MemoryEventStore
  let trajectoryStore: TrajectoryStore
  let milkie: Milkie
  let trajectory: Trajectory
  let lastResult: AgentResult

  const contextId = `ctx-repair-pathD-${Date.now()}`
  const goal = '为报修工单定位负责人'
  // One turn carries all four entity keywords; the pinned-filtered resolver
  // disambiguates each level out of it.
  const INPUT = '总部 主楼 IT网络部 王芳'

  async function sendTurn(input: string): Promise<AgentResult> {
    return milkie.invoke({ agentId: 'repair-ticketing', goal, input, contextId })
  }

  beforeAll(async () => {
    eventStore = new MemoryEventStore()
    trajectoryStore = new TrajectoryStore({ jsonlDir: './test-output/trajectories' })
    milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore,
      trajectoryStore,
      gateway: new HerSlotGateway(),
      tools,
    })
    milkie.registerAgent(repairAgentConfig)

    lastResult = await sendTurn(INPUT)  // lookup→commit ×4 in one tool-loop

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
    const wm = await latestWorkingMemory(eventStore, lastResult.agentRunId)
    expect(wm).toMatchObject({
      site:       'S01',
      building:   'B01',
      department: 'D03',
      assignee:   'E008',
    })
  })

  it('the single state runs the whole slot set to a completed turn (AC8)', () => {
    // #175: no SLOTS_COMPLETE event, no fsm.transition to a `confirming` state —
    // all four levels resolve within one autonomous loop and the turn completes.
    expect(lastResult.status).toBe('completed')
    const transitionSpans = trajectory.spans.filter(s => s.name === 'fsm.transition')
    expect(transitionSpans).toEqual([])  // no business transitions remain
  })
})
