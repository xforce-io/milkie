// Repair-ticketing agent — the web UI layer over the portable hierarchical
// entity resolver (#162 / S-011 Path D), migrated to the #175 "lightweight tier".
//
// #175 de-core: the old three-state business FSM
//   collecting_entities → collecting_description → emit_ticket (action) → completed
// is GONE. The core runtime no longer steps user-state → user-state; a run drives
// one authored state to its outcome. So the whole journey now lives in a SINGLE
// autonomous `llm` state whose tool-loop does everything, exactly the
// docs/design/175 §6 "structured dialogue without a state machine" recipe:
//
//   • 软顺序 / 阶段聚焦   → systemPrompt (软引导，非硬拓扑)
//   • 槽位完整性          → tool param schema + the resolver's own validation
//                          (硬地板，本地规则校验) — lookup_entity / commit_entity
//   • 跨 turn 阶段记忆     → working memory (the four level ids + description live
//                          in WM and survive across turns within the session)
//   • 一两条硬门 (Y→X)    → action precondition: assemble_ticket REFUSES unless all
//                          four levels AND the description are confirmed in WM
//                          ("未报价不得维修" 的等价物：未集齐不得开单)
//
// There is no ctx.emit, no `on:` map, no business event. "All slots filled" is a
// plain fact read back from WM by the assemble_ticket precondition.

import type { AgentConfig } from '../../../src/types/agent.js'
import type { ToolContext, ToolDefinition } from '../../../src/types/tool.js'
import type { EntityResolver } from '../resolver/EntityResolver.js'
import { makeEntityResolverTools, makeResolveEntitiesTool } from './tools/entity-resolver.js'

/** Ordered hierarchy levels that must all be confirmed before a ticket can open. */
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
 * Fields required in working memory before a ticket may be opened — the four
 * hierarchy levels plus the free-text description. The assemble_ticket
 * precondition checks exactly these.
 */
const TICKET_FIELDS = [...REQUIRED_SLOTS, 'description'] as const

/**
 * Read the confirmed ticket fields out of WM. Returns the field map when ALL are
 * present and non-empty, or the list of missing field names otherwise. This is
 * the shared body of the assemble_ticket action precondition: it is a pure WM
 * read, the "local rule" half of docs/design/175 §6 — no LLM, no ontology lookup.
 */
function readTicketFields(
  ctx: ToolContext,
): { ok: true; fields: Record<(typeof TICKET_FIELDS)[number], string> } | { ok: false; missing: string[] } {
  const fields = {} as Record<(typeof TICKET_FIELDS)[number], string>
  const missing: string[] = []
  for (const key of TICKET_FIELDS) {
    const value = ctx.workingMemory.get(key)
    if (value == null || value === '') {
      missing.push(key)
    } else {
      fields[key] = String(value)
    }
  }
  return missing.length === 0 ? { ok: true, fields } : { ok: false, missing }
}

/**
 * `commit_description` — records the user's free-text fault description into WM.
 * The text is read from `ctx.currentTurn` (the trusted runtime turn), NOT a tool
 * parameter, mirroring the HER adapter's trust boundary, so the stored
 * description is the user's verbatim words — not a model paraphrase.
 *
 * #175: it no longer fires a business event. Writing `description` into WM is all
 * it does; the downstream assemble_ticket precondition reads that fact back.
 *
 * #185: optional `description` param for the one-shot case. When the user packs
 * levels AND the fault into a SINGLE turn, taking ctx.currentTurn verbatim would
 * contaminate the description with the level原话. So the model may pass the clean
 * fault substring as `description`; it must be a substring of the user's words, not
 * a paraphrase. The param is OPT-IN — absent or blank, behavior is unchanged: the
 * verbatim turn is used, preserving the #162/#164 "user's words, not a model
 * paraphrase" trust boundary for the normal multi-turn flow.
 */
export function makeCommitDescriptionTool(): ToolDefinition {
  return {
    name: 'commit_description',
    description:
      '记录用户对故障现象的自由文本描述。多轮场景下你无需传参，系统自动采用用户本轮原话；' +
      '仅当用户在同一轮里把「报修对象 + 故障现象」一起说了、需要把纯故障部分单独拎出来时，' +
      '才传入 description（必须是用户原话中描述故障的子串，不要改写、不要包含站点/楼宇/部门/负责人）。' +
      '记录后即可调用 assemble_ticket 生成工单。',
    inputSchema: {
      type: 'object',
      properties: {
        description: {
          type: 'string',
          description:
            '可选，仅单轮合并场景使用：用户原话中描述故障现象的子串（不要改写，不要含层级名称）。' +
            '缺省则自动采用用户本轮原话。',
        },
      },
    },
    handler: async (input: unknown, ctx: ToolContext): Promise<unknown> => {
      const param = (input as { description?: string } | null | undefined)?.description?.trim()
      const description = param && param.length > 0 ? param : (ctx.currentTurn ?? '').trim()
      ctx.workingMemory.set('description', description)
      return { description }
    },
  }
}

/** Returned by assemble_ticket when its precondition is unmet (#175 action
 *  precondition: "未集齐不得开单"). The LLM sees which fields are still missing and
 *  keeps collecting instead of opening a ticket. */
export interface AssembleTicketBlocked {
  preconditionFailed: 'incomplete_fields'
  missing:            string[]
  message:            string
}

/**
 * `assemble_ticket` — the lightweight-tier replacement for the old `emit_ticket`
 * action STATE. It is now an ordinary LLM-callable tool guarded by an ACTION
 * PRECONDITION (docs/design/175 §6, "一两条硬门"): it refuses to assemble a ticket
 * unless every required field is already confirmed in working memory — the four
 * hierarchy levels and the description. This is the "未报价不得维修" analog: the hard
 * gate lives in the tool, not in an FSM edge.
 *
 * On a met precondition it assembles the ticket deterministically (no LLM, pure
 * `assembleTicket`), stores it in WM, and returns it as a JSON string — which the
 * model relays as its final turn text, so the live `milkie.invoke` output is the
 * ticket. On an unmet precondition it returns a structured block telling the model
 * which fields are missing; it does NOT throw (a throw would fail the whole run —
 * #175: an action handler error is the `error` signal). A precondition miss is a
 * recoverable nudge to keep collecting, not a run failure.
 */
export function makeAssembleTicketTool(): ToolDefinition {
  return {
    name: 'assemble_ticket',
    description:
      '从工作记忆中已确认的字段确定性拼装报修工单（纯函数，无 LLM）。前置条件：站点 / 楼宇 / 部门 / ' +
      '负责人四级实体与故障描述必须全部确认。若有缺失会返回 preconditionFailed 与 missing 列表——' +
      '此时不要重复调用，应先补齐缺失字段（继续 lookup_entity/commit_entity 或请用户补充描述）。',
    inputSchema: { type: 'object', properties: {} },
    handler: async (_input: unknown, ctx: ToolContext): Promise<unknown> => {
      const read = readTicketFields(ctx)
      if (!read.ok) {
        // Action precondition unmet → structured block, NOT a throw.
        const blocked: AssembleTicketBlocked = {
          preconditionFailed: 'incomplete_fields',
          missing:            read.missing,
          message:            `尚不能生成工单：缺少已确认字段 ${read.missing.join(', ')}。请先补齐后再调用。`,
        }
        return blocked
      }
      const ticket = assembleTicket(read.fields)
      ctx.workingMemory.set('ticket', ticket)
      return JSON.stringify(ticket)
    },
  }
}

/**
 * Wire the full tool set for the example over a loaded resolver: the HER pair
 * (`lookup_entity` + `commit_entity`), `commit_description`, and `assemble_ticket`.
 * Shared by the server and the e2e so both drive the identical agent.
 */
export function buildRepairTicketingTools(resolver: EntityResolver): ToolDefinition[] {
  return [
    // #180: resolve_entities (recall + fast-path) is the primary entity tool; the
    // lower-level lookup_entity/commit_entity pair stays registered — commit_entity
    // is the model's step-2 "pick from candidates", and both remain the resolver
    // adapter's tested contract (#166).
    makeResolveEntitiesTool(resolver, [...REQUIRED_SLOTS]),
    ...makeEntityResolverTools(resolver, [...REQUIRED_SLOTS]),
    makeCommitDescriptionTool(),
    makeAssembleTicketTool(),
  ]
}

export const repairTicketingAgentConfig: AgentConfig = {
  agentId:      'repair-ticketing',
  version:      '2.0.0',
  systemPrompt:
    `你是报修工单助手，在一个自主对话中按以下顺序协助用户（软引导，非强制状态机）：
1. 逐级定位报修对象（站点 → 楼宇 → 部门 → 负责人）。每轮先调用 resolve_entities——系统会用用户本轮原话自动检索，并把能【唯一确定】的层级直接确认（无需你指定 level，也无需传参）。然后按其返回处理：
   - committed：已自动确认的层级，无需再操作。
   - needsSelection：某一级有多个候选。若能从上下文或对话历史判断是哪一个，就用 commit_entity 选定那个候选的 id（id 必须来自候选，不得臆造）；若无法判断，就把 message 转述给用户，请其在候选中选择。
   - notFound：该层级本轮还无法确定。请按 message 请用户补充这一层级。
2. 四级全部确认后，请用户用一句话描述故障现象（位置 + 问题）。用户描述后调用 commit_description（描述内容由系统自动采集，无需传参）。
3. 四级实体与描述都确认后，调用 assemble_ticket 生成结构化工单，并把工单原样回复用户。
若用户在同一轮里就把报修对象与故障现象一起说全了，直接在本轮依次走完：resolve_entities / commit_entity 定位四级 → commit_description（此时把其中描述故障的子串作为 description 参数传入，不要把整句原话连同站点/楼宇/部门/负责人一起当描述）→ assemble_ticket，不要再停下来追问描述。
注意：assemble_ticket 有前置条件——必须先集齐四级实体与描述；若它返回 preconditionFailed，请先补齐 missing 列出的字段，不要重复空调用。`,
  // #175: a single autonomous llm state. No `on:` edges, no business events.
  // #180: resolve_entities does deterministic recall + fast-path commit; commit_entity
  // is the model's step-2 pick from candidates; commit_description + assemble_ticket
  // close the journey. lookup_entity is intentionally NOT exposed here (resolve_entities
  // supersedes it) though it stays registered for the adapter contract/tests.
  fsm: {
    states: [
      {
        name:  'repair',
        type:  'llm',
        tools: ['resolve_entities', 'commit_entity', 'commit_description', 'assemble_ticket'],
      },
    ],
  },
  model: { provider: 'volcengine', model: 'doubao-seed-2.0-lite', adapter: 'openai-compatible' },
}
