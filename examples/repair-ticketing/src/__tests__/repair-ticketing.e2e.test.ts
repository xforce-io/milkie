/**
 * Issue #162 — repair-ticketing example: the full three-state flow as an e2e.
 *
 * This is the coverage the #166 adapter test does NOT provide: the end-to-end
 * journey through all three FSM states —
 *   collecting_entities → collecting_description → emit_ticket (action) → completed
 * — driven by a deterministic stub gateway, asserting the ticket assembled by the
 * pure-function action handler.
 *
 * Two #162 acceptance points are pinned here:
 *   1. emit_ticket is an `action` state whose handler assembles the ticket
 *      deterministically from confirmed WM — never LLM-authored.
 *   2. The ticket is read from the LIVE `milkie.invoke` return value (result.output),
 *      NOT from a checkpoint: emit_ticket → completed is a terminal turn, and #172
 *      does not persist a checkpoint for terminal turns, so a checkpoint read would
 *      drop the final state.
 *
 * The error-path block at the bottom exercises every CommitOutput status the
 * resolver can return (alias / missing / ambiguous / unknown / invalid_selection /
 * corrected). These overlap with the #166 adapter test by design — #162's
 * acceptance asks for them in this example's e2e, so they live here too to keep
 * #162 self-contained.
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import type { AgentResult, Message } from '../../../../src/types/common.js'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../../../src/types/model.js'
import type { ToolContext, ToolDefinition } from '../../../../src/types/tool.js'
import { Milkie } from '../../../../src/runtime/Milkie.js'
import { MemoryStore } from '../../../../src/store/MemoryStore.js'
import { MemoryEventStore } from '../../../../src/trace/MemoryEventStore.js'
import { WorkingMemory } from '../../../../src/store/WorkingMemory.js'

import { EntityResolver, type Schema } from '../../resolver/EntityResolver.js'
import { assembleTicket, buildRepairTicketingTools, repairTicketingAgentConfig } from '../agent.js'

// ─────────────────────────────── Fixtures ────────────────────────────────────

const schema = JSON.parse(
  readFileSync(join(__dirname, '../../resolver/schema.json'), 'utf8'),
) as Schema
const csv = readFileSync(join(__dirname, '../../resolver/data.csv'), 'utf8')

const resolver = EntityResolver.load(schema, csv)
const tools = buildRepairTicketingTools(resolver)
const lookupTool = tools.find(t => t.name === 'lookup_entity')!
const commitTool = tools.find(t => t.name === 'commit_entity')!

// ─────────────────────────── Pure-function assembly ──────────────────────────

describe('assembleTicket (pure, LLM-free)', () => {
  it('assembles a fully deterministic ticket from confirmed fields (#162)', () => {
    const ticket = assembleTicket(
      { site: 'S01', building: 'B01', department: 'D03', assignee: 'E008', description: '投影仪无法开机' },
      () => '2026-06-16T00:00:00.000Z',  // injected clock → exact, assertable createdAt
    )
    expect(ticket).toEqual({
      ticketId:    'TKT-S01-B01-D03-E008',
      site:        'S01',
      building:    'B01',
      department:  'D03',
      assignee:    'E008',
      description: '投影仪无法开机',
      createdAt:   '2026-06-16T00:00:00.000Z',
    })
  })
})

// ─────────────────────────── Deterministic gateway ───────────────────────────

/**
 * Drives the whole flow with no live model. Per the active FSM state (tracked by
 * counting committed entity levels) it issues `lookup_entity` then `commit_entity`
 * for each of the four levels, then `commit_description` once entities are done.
 * Selections come from the resolver's own suggestion — the model never invents an
 * id, and never supplies the utterance/description (the tools read those from the
 * runtime turn).
 */
class RepairTicketGateway implements IModelGateway {
  private readonly levelOrder = ['site', 'building', 'department', 'assignee']
  private committed = 0

  async complete(request: ModelRequest): Promise<ModelResponse> {
    const messages = request.messages
    const last = messages[messages.length - 1]
    const producedBy = last && last.role === 'tool' ? this.precedingToolUse(messages) : undefined
    const result = last && last.role === 'tool' ? this.parseToolResult(last) : null

    if (producedBy === 'lookup_entity') {
      const level = this.levelOrder[this.committed]!
      const opts = (result?.['options'] as string[] | undefined) ?? []
      const selected = (result?.['suggested'] as string | null) ?? opts[0]
      return this.tool(`commit-${level}`, 'commit_entity', { selected, context: { level } })
    }
    if (producedBy === 'commit_entity') {
      this.committed++
      return this.text(
        this.committed >= this.levelOrder.length
          ? '已登记到负责人，请用一句话描述故障现象。'
          : '已记录，请继续。',
      )
    }
    if (producedBy === 'commit_description') {
      return this.text('已记录故障描述。')
    }

    // Fresh user turn (no preceding tool result).
    if (this.committed < this.levelOrder.length) {
      const level = this.levelOrder[this.committed]!
      return this.tool(`lookup-${level}`, 'lookup_entity', { context: { level } })
    }
    // All four levels confirmed → this turn carries the free-text description.
    return this.tool('commit-desc', 'commit_description', {})
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

// ─────────────────────────── Happy path (full FSM) ───────────────────────────

describe('repair-ticketing — happy path through emit_ticket (FSM e2e)', () => {
  const DESCRIPTION = '三楼会议室的投影仪无法开机'
  const contextId = `ctx-repair-ticket-${Date.now()}`
  let milkie: Milkie
  let lastResult: AgentResult

  async function sendTurn(input: string): Promise<AgentResult> {
    return milkie.invoke({ agentId: 'repair-ticketing', goal: '登记一张报修工单', input, contextId })
  }

  beforeAll(async () => {
    milkie = new Milkie({
      stateStore: new MemoryStore(),
      eventStore: new MemoryEventStore(),
      gateway:    new RepairTicketGateway(),
    })
    for (const tool of tools) milkie.registerTool(tool)
    milkie.registerAgent(repairTicketingAgentConfig)

    await sendTurn('总部')          // → site S01
    await sendTurn('主楼')          // → building B01
    await sendTurn('IT网络部')      // → department D03
    await sendTurn('王芳')          // → assignee E008 → SLOTS_COMPLETE → collecting_description
    lastResult = await sendTurn(DESCRIPTION)  // → commit_description → emit_ticket → completed
  }, 60_000)

  it('reaches a completed terminal turn', () => {
    expect(lastResult.status).toBe('completed')
  })

  it('emits the ticket on the live invoke output, deterministically assembled from WM (#162)', () => {
    // Read the ticket from the LIVE return value — not a checkpoint (#172: the
    // terminal turn persists none).
    const ticket = JSON.parse(lastResult.output)
    expect(ticket).toMatchObject({
      ticketId:    'TKT-S01-B01-D03-E008',
      site:        'S01',
      building:    'B01',
      department:  'D03',
      assignee:    'E008',
      description: DESCRIPTION,
    })
    // createdAt is a system timestamp — assert its shape, not an exact value.
    expect(ticket.createdAt).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/)
  })
})

// ─────────────────────────── Error paths (resolver statuses) ──────────────────

function makeCtx(currentTurn: string, preset: Record<string, string> = {}): {
  ctx: ToolContext; emitted: string[]; wm: WorkingMemory
} {
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

const lookup = (tool: ToolDefinition, level: string, ctx: ToolContext) =>
  tool.handler({ context: { level } }, ctx) as Promise<{ options: string[]; suggested: string | null }>

const commit = (tool: ToolDefinition, selected: string, level: string, ctx: ToolContext) =>
  tool.handler({ selected, context: { level } }, ctx) as Promise<{
    resolved?: { id: string }; corrected?: Record<string, string>; validationError?: string
  }>

describe('repair-ticketing — every resolver status reachable through the tools (#162)', () => {
  it('alias match: a department alias resolves and is suggested', async () => {
    const { ctx } = makeCtx('硬件部', { site: 'S01', building: 'B01' })
    const out = await lookup(lookupTool, 'department', ctx)
    expect(out.options).toContain('D02')   // 硬件部 = alias of IT硬件组 (D02)
    expect(out.suggested).toBe('D02')
  })

  it('missing-level: an under-selection at the wrong level is rejected', async () => {
    const { ctx, wm } = makeCtx('总部')
    const out = await commit(commitTool, 'B01', 'site', ctx)  // B01 is a building, not a site
    expect(out.validationError).toBeDefined()
    expect(wm.get('site')).toBeUndefined()
  })

  it('ambiguous: multiple equally-scored candidates yield no suggestion', async () => {
    const { ctx } = makeCtx('张', { site: 'S01', building: 'B01', department: 'D03' })
    const out = await lookup(lookupTool, 'assignee', ctx)
    expect(out.options.length).toBeGreaterThanOrEqual(2)  // 张伟(E007) + 张亮(E009)
    expect(out.suggested).toBeNull()
  })

  it('unknown: an id that exists at no level is rejected', async () => {
    const { ctx, wm } = makeCtx('总部')
    const out = await commit(commitTool, 'ZZZ', 'site', ctx)
    expect(out.validationError).toBeDefined()
    expect(wm.get('site')).toBeUndefined()
  })

  it('invalid_selection: a selection conflicting with the pinned branch and uncorrectable is rejected', async () => {
    // pinned 总部/东楼 (S01/B02); E020 lives under 分部 (S02) with no same-label
    // sibling under the pinned branch → no correction possible.
    const { ctx, wm } = makeCtx('赵明', { site: 'S01', building: 'B02' })
    const out = await commit(commitTool, 'E020', 'assignee', ctx)
    expect(out.validationError).toBeDefined()
    expect(wm.get('assignee')).toBeUndefined()
  })

  it('corrected: a selection conflicting with the pinned branch auto-corrects to the same-label sibling', async () => {
    // pinned 总部/主楼 (S01/B01); E012 (张伟, under B02/D07) auto-corrects to the
    // same-label E007 (张伟, under B01/D03).
    const { ctx, wm } = makeCtx('张伟', { site: 'S01', building: 'B01' })
    const out = await commit(commitTool, 'E012', 'assignee', ctx)
    expect(out.resolved?.id).toBe('E007')
    expect(out.corrected).toMatchObject({ department: 'D03' })
    expect(wm.get('assignee')).toBe('E007')
  })
})
