// Repair-ticketing agent — the web UI layer over the portable hierarchical
// entity resolver (#162 / S-011 Path D).
//
// Three-state FSM:
//   collecting_entities  (llm)    lookup_entity + commit_entity resolve the four
//                                 hierarchy levels in-process; SLOTS_COMPLETE →
//                                 collecting_description.
//   collecting_description (llm)  one NON-HER tool, commit_description, records
//                                 the user's free-text fault description into WM
//                                 and fires DESCRIPTION_READY → emit_ticket.
//   emit_ticket          (action) a pure, LLM-free handler that assembles the
//                                 final ticket from the already-confirmed WM
//                                 fields; DONE → completed (terminal).
//
// Why collecting_description carries a tool at all: an `llm` state can only fire
// a named business event (DESCRIPTION_READY) from a tool's `ctx.emit`, and the
// emit_ticket handler reads `description` from WM — so *something* must both
// write the description and emit. commit_description is that something. It is not
// a hierarchical-entity (HER) tool, honoring "no HER tools" while keeping the
// state self-consistent. Like the HER tools, it takes the description from the
// runtime turn (`ctx.currentTurn`), never from an LLM-supplied parameter, so the
// stored description is the user's verbatim words — not a model paraphrase.

import type { AgentConfig } from '../../../src/types/agent.js'
import type { ToolContext, ToolDefinition } from '../../../src/types/tool.js'
import type { EntityResolver } from '../resolver/EntityResolver.js'
import { makeEntityResolverTools } from './tools/entity-resolver.js'

/** Ordered hierarchy levels that must all be confirmed before SLOTS_COMPLETE. */
export const REQUIRED_SLOTS = ['site', 'building', 'department', 'assignee'] as const

/**
 * The structured repair ticket. Every field is derived deterministically from
 * confirmed working-memory state — the LLM never authors this object (#162: the
 * final ticket is assembled from verified WM, not invented by the model, so it
 * can be asserted in tests).
 */
export interface Ticket {
  ticketId:    string
  site:        string
  building:    string
  department:  string
  assignee:    string
  description: string
  createdAt:   string
}

/**
 * Pure-function ticket assembly. No LLM, no I/O, no hidden state — given the same
 * confirmed fields and the same clock it returns the same ticket, so callers can
 * assert the whole object exactly (inject `now` in tests for a fixed timestamp).
 *
 * `ticketId` is a deterministic function of the four resolved entity ids, so two
 * tickets for the same target collide by design — fine for a demo, and it keeps
 * the id assertable without a counter or RNG.
 */
export function assembleTicket(
  fields: { site: string; building: string; department: string; assignee: string; description: string },
  now: () => string = () => new Date().toISOString(),
): Ticket {
  return {
    ticketId:    `TKT-${fields.site}-${fields.building}-${fields.department}-${fields.assignee}`,
    site:        fields.site,
    building:    fields.building,
    department:  fields.department,
    assignee:    fields.assignee,
    description: fields.description,
    createdAt:   now(),
  }
}

/**
 * `commit_description` — the lone tool of `collecting_description`. Records the
 * free-text fault description into WM and fires DESCRIPTION_READY. The text is
 * read from `ctx.currentTurn` (the trusted runtime turn), NOT a tool parameter,
 * mirroring the HER adapter's trust boundary.
 */
export function makeCommitDescriptionTool(): ToolDefinition {
  return {
    name: 'commit_description',
    description:
      '记录用户对故障现象的自由文本描述。系统会自动采用用户本轮的原始输入作为描述内容——' +
      '你无需传入任何参数，只需在用户给出描述后调用本工具。确认后进入工单生成阶段。',
    inputSchema: { type: 'object', properties: {} },
    handler: async (_input: unknown, ctx: ToolContext): Promise<unknown> => {
      const description = (ctx.currentTurn ?? '').trim()
      ctx.workingMemory.set('description', description)
      ctx.emit('DESCRIPTION_READY')
      return { description }
    },
  }
}

/**
 * `assemble_ticket` — the handler for the `emit_ticket` action state. Reads the
 * confirmed fields from WM and assembles the ticket deterministically (no LLM).
 *
 * The handler's return value becomes the run's `lastTextOutput` →
 * `AgentResult.output`. That is deliberate: `emit_ticket → completed` is a
 * terminal turn, which per #172 does NOT persist a checkpoint, so the ticket is
 * only reliably available from the live `milkie.invoke` return value. Callers
 * (server, e2e) read it there — never from a checkpoint.
 */
export function makeAssembleTicketTool(): ToolDefinition {
  return {
    name: 'assemble_ticket',
    description: 'Action 处理器：从工作记忆已确认字段确定性拼装报修工单（纯函数，无 LLM）。',
    inputSchema: { type: 'object', properties: {} },
    handler: async (_input: unknown, ctx: ToolContext): Promise<unknown> => {
      const get = (key: string): string => {
        const value = ctx.workingMemory.get(key)
        if (value == null || value === '') {
          throw new Error(`assemble_ticket: working memory missing confirmed field "${key}"`)
        }
        return String(value)
      }
      const ticket = assembleTicket({
        site:        get('site'),
        building:    get('building'),
        department:  get('department'),
        assignee:    get('assignee'),
        description: get('description'),
      })
      ctx.workingMemory.set('ticket', ticket)
      return JSON.stringify(ticket)
    },
  }
}

/**
 * Wire the full tool set for the example over a loaded resolver: the HER pair
 * (`lookup_entity` + `commit_entity`), `commit_description`, and the
 * `assemble_ticket` action handler. Shared by the server and the e2e so both
 * drive the identical agent.
 */
export function buildRepairTicketingTools(resolver: EntityResolver): ToolDefinition[] {
  return [
    ...makeEntityResolverTools(resolver, [...REQUIRED_SLOTS]),
    makeCommitDescriptionTool(),
    makeAssembleTicketTool(),
  ]
}

export const repairTicketingAgentConfig: AgentConfig = {
  agentId:      'repair-ticketing',
  version:      '1.0.0',
  systemPrompt:
    `你是报修工单助手，按以下流程协助用户：
1. 逐级定位报修对象（站点 → 楼宇 → 部门 → 负责人）。对每一级：先调用 lookup_entity 检索候选，再调用 commit_entity 确认其中一个候选。系统会自动以用户本轮原话作为查询，你只需指定 level。
2. 四级全部确认后，请用户用一句话描述故障现象（位置 + 问题）。用户描述后调用 commit_description（描述内容由系统自动采集，无需传参）。
3. 系统随后自动生成结构化工单。`,
  fsm: {
    states: [
      {
        name:  'collecting_entities',
        type:  'llm',
        tools: ['lookup_entity', 'commit_entity'],
        on:    { SLOTS_COMPLETE: 'collecting_description' },
      },
      {
        name:         'collecting_description',
        type:         'llm',
        tools:        ['commit_description'],
        instructions: '请用户用一句话描述故障现象（所在位置 + 具体问题）。用户给出描述后调用 commit_description。',
        on:           { DESCRIPTION_READY: 'emit_ticket' },
      },
      {
        // Action state: the handler assembles the ticket deterministically and
        // auto-DONEs (no ctx.emit) → completed.
        name:    'emit_ticket',
        type:    'action',
        handler: 'assemble_ticket',
        on:      { DONE: 'completed' },
      },
      {
        // Terminal: executeFSM breaks on entry (non-llm + terminal), so no extra
        // LLM call happens after the ticket is assembled.
        name:     'completed',
        type:     'action',
        terminal: true,
      },
    ],
  },
  model: { provider: 'volcengine', model: 'doubao-seed-2.0-lite', adapter: 'openai-compatible' },
}
