// Milkie HER adapter (#166 / S-011 Path D).
//
// Wires the portable hierarchical entity resolver core (#167) into a pair of
// milkie `ToolDefinition`s so the repair-ticketing example can resolve
// hierarchical entities IN-PROCESS inside a single autonomous loop — no
// subprocess, no CLI spawn.
//
// #175 de-core: this adapter no longer fires a business event (it used to
// `ctx.emit('SLOTS_COMPLETE')`). There is no multi-state FSM anymore — slot
// completeness is a plain fact in working memory, read back by the downstream
// `assemble_ticket` precondition (see src/agent.ts). The runtime offers no
// `ctx.emit`; "all slots filled" is just `every(level → WM has it)`.
//
// Trust boundary: the LLM never supplies the raw utterance or the pinned ancestor
// context through tool parameters. The adapter reads:
//   • the utterance from `ctx.currentTurn` (runtime-injected, stable per turn), and
//   • `pinned` from `ctx.workingMemory` (already-committed level ids),
// both trusted runtime state. The LLM may only choose the target `level`, an
// optional `sessionHint`, and (for commit) the `selected` id.

import type { ToolContext, ToolDefinition } from '../../../../src/types/tool.js'
import type { EntityResolver, ResolvedEntity } from '../../resolver/EntityResolver.js'

interface LookupInput {
  context?: { level?: string; sessionHint?: string }
}

interface CommitInput {
  selected?: string
  context?: { level?: string }
}

/**
 * `commit_entity` output contract.
 *
 * On success the adapter returns the `resolved` entity. When the resolver
 * auto-corrects a selection that conflicts with the pinned ancestor branch, it
 * also returns `corrected` — a map of the level name(s) that were overridden to
 * their corrected id (mirrors `CommitOutput.correctedLevels`). It is a
 * `Record<string, string>`, NOT a plain string: a single correction can touch a
 * level other than the committed one (e.g. committing `assignee` may report a
 * corrected `department`). On any non-success status the adapter returns
 * `{ validationError }` instead.
 */
interface CommitToolOutput {
  resolved?: ResolvedEntity
  corrected?: Record<string, string>
  validationError?: string
}

/**
 * Build the `lookup_entity` / `commit_entity` tool pair over an already-loaded
 * resolver.
 *
 * @param resolver      constructed once at startup via `EntityResolver.load(...)`;
 *                      handlers reuse it and never re-parse schema/CSV.
 * @param requiredSlots ordered level names whose already-committed members are the
 *                      `pinned` ancestor branch for the next lookup/commit
 *                      (e.g. ['site','building','department','assignee']).
 */
export function makeEntityResolverTools(
  resolver: EntityResolver,
  requiredSlots: string[],
): ToolDefinition[] {
  // pinned = the subset of required levels already committed in WM. Read purely
  // from runtime state — never from LLM tool params.
  const pinnedFromWM = (ctx: ToolContext): Record<string, string> => {
    const pinned: Record<string, string> = {}
    for (const level of requiredSlots) {
      const value = ctx.workingMemory.get(level)
      if (value != null && value !== '') pinned[level] = String(value)
    }
    return pinned
  }

  const lookupEntity: ToolDefinition = {
    name:        'lookup_entity',
    description:
      '在层级实体词典中检索目标层级（level）的候选实体。系统会自动使用本轮用户的原始输入作为查询语句，' +
      '并依据已确认的上级实体过滤分支——你只需指定 level（可选 sessionHint）。返回 candidates / options / suggested。',
    inputSchema: {
      type: 'object',
      properties: {
        context: {
          type:       'object',
          properties: {
            level:       { type: 'string', description: '目标层级名称，如 site / building / department / assignee' },
            sessionHint: { type: 'string', description: '可选：跨轮会话提示' },
          },
          required: ['level'],
        },
      },
      required: ['context'],
    },
    handler: async (input: unknown, ctx: ToolContext): Promise<unknown> => {
      const { context } = (input ?? {}) as LookupInput
      // Utterance comes from the runtime turn, NOT from the LLM tool params.
      const utterance = ctx.currentTurn ?? ''
      return resolver.lookup({
        utterance,
        level:       context?.level,
        pinned:      pinnedFromWM(ctx),
        sessionHint: context?.sessionHint,
      })
    },
  }

  const commitEntity: ToolDefinition = {
    name:        'commit_entity',
    description:
      '确认在目标层级（level）选定的实体。selected 必须是上一次 lookup_entity 返回的 options 或 suggested 中的值。' +
      '系统依据已确认的上级实体校验该选择；确认成功后写入工作记忆。四级（站点/楼宇/部门/负责人）全部确认后即可进入下一步。',
    inputSchema: {
      type: 'object',
      properties: {
        selected: { type: 'string', description: '选定的实体 id（来自上一次 lookup 的 options/suggested）' },
        context:  {
          type:       'object',
          properties: { level: { type: 'string', description: '目标层级名称' } },
          required:   ['level'],
        },
      },
      required: ['selected', 'context'],
    },
    handler: async (input: unknown, ctx: ToolContext): Promise<unknown> => {
      const { selected, context } = (input ?? {}) as CommitInput
      const level = context?.level
      const result = resolver.commit({
        selected: selected ?? '',
        level,
        pinned:   pinnedFromWM(ctx),
      })

      if (result.status === 'complete' || result.status === 'corrected') {
        // WM key = level name; value = the resolved entity id. No business event:
        // slot completeness is read back from WM by the assemble_ticket
        // precondition, not pushed via a (now-removed) ctx.emit (#175).
        const key = level ?? result.resolved.path.length.toString()
        ctx.workingMemory.set(key, result.resolved.id)

        // `corrected` is a Record<string,string> (level → corrected id), per
        // CommitToolOutput — see the type doc for why it is not a plain string.
        const output: CommitToolOutput =
          result.status === 'corrected'
            ? { resolved: result.resolved, corrected: result.correctedLevels }
            : { resolved: result.resolved }
        return output
      }

      // invalid_selection / missing / ambiguous / unknown → no WM write.
      const errorOutput: CommitToolOutput = { validationError: result.message }
      return errorOutput
    },
  }

  return [lookupEntity, commitEntity]
}
