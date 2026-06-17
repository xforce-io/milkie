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
import type { RecallConfig } from '../../resolver/recall.js'
import { repairMessages } from '../messages.js'

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
/**
 * Coerce a tool's `context` argument into an object. Some models (e.g. DeepSeek)
 * serialize nested-object parameters as a JSON *string* instead of a JSON object;
 * left unparsed, `context.level` would read `undefined` and the slot would be
 * written under the wrong key. Parse a string form defensively; pass an object
 * through; anything else → undefined.
 */
function coerceContext<T extends object>(raw: unknown): T | undefined {
  if (raw == null) return undefined
  if (typeof raw === 'string') {
    try { return JSON.parse(raw) as T } catch { return undefined }
  }
  return typeof raw === 'object' ? raw as T : undefined
}

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
      const context = coerceContext<NonNullable<LookupInput['context']>>((input as LookupInput)?.context)
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
      const { selected } = (input ?? {}) as CommitInput
      const context = coerceContext<NonNullable<CommitInput['context']>>((input as CommitInput)?.context)
      const level = context?.level
      // The WM slot key IS the level name. Without a level we cannot key the slot
      // correctly — fail loudly rather than silently writing under a numeric
      // fallback key (which leaves requiredSlots permanently incomplete).
      if (!level) {
        const errorOutput: CommitToolOutput = { validationError: 'commit_entity: missing level in context' }
        return errorOutput
      }
      const result = resolver.commit({
        selected: selected ?? '',
        level,
        pinned:   pinnedFromWM(ctx),
      })

      if (result.status === 'complete' || result.status === 'corrected') {
        // WM key = level name; value = the resolved entity id. `level` is
        // guaranteed present (early-returned above, #178 — no numeric fallback).
        // No business event: slot completeness is read back from WM by the
        // assemble_ticket precondition, not pushed via a (now-removed)
        // ctx.emit (#175).
        ctx.workingMemory.set(level, result.resolved.id)

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

/**
 * `resolve_entities` — the #180 "step 1 + fast-path" orchestration tool. One call
 * per user turn does the deterministic work the model used to fumble:
 *
 *   for each still-unfilled level, in order (站点→楼宇→部门→负责人):
 *     • run the fusion recall (resolver.recall) over this turn's utterance, branch-
 *       filtered by the already-committed ancestors;
 *     • if it has a DECISIVE unique winner → commit it straight to WM (no LLM) and
 *       move to the next level;
 *     • else stop: report the level as `needsSelection` (multiple candidates — the
 *       model may pick with commit_entity if context disambiguates, else clarify)
 *       or `notFound` (none — prompt the user for this level).
 *
 * The LLM never picks a level and never invents an id: decisive levels are
 * committed deterministically; ambiguous ones hand the model REAL candidates to
 * choose from. No business event (#175) — completeness is read back from WM by the
 * assemble_ticket precondition. User-facing wording comes from messages.ts.
 */
export function makeResolveEntitiesTool(
  resolver: EntityResolver,
  requiredSlots: string[],
  recallConfig?: RecallConfig,
): ToolDefinition {
  const pinnedFromWM = (ctx: ToolContext): Record<string, string> => {
    const pinned: Record<string, string> = {}
    for (const level of requiredSlots) {
      const value = ctx.workingMemory.get(level)
      if (value != null && value !== '') pinned[level] = String(value)
    }
    return pinned
  }
  const firstUnfilled = (ctx: ToolContext): string | undefined =>
    requiredSlots.find(l => {
      const v = ctx.workingMemory.get(l)
      return v == null || v === ''
    })

  return {
    name: 'resolve_entities',
    description:
      '用用户本轮原话自动逐级检索并提交能【唯一确定】的层级（站点→楼宇→部门→负责人，系统自动取原话为查询，你无需传参）。' +
      '返回 committed（已自动确认）、needsSelection（同级多个候选，需判断）、notFound（该层级尚未给出）、message（可向用户转述的提示）。' +
      '处理建议：needsSelection 时若能从上下文/历史判断是哪一个，用 commit_entity 选定；否则把 message 转述给用户澄清。notFound 时请用户补充该层级。每轮先调用本工具。',
    inputSchema: { type: 'object', properties: {} },
    handler: async (_input: unknown, ctx: ToolContext): Promise<unknown> => {
      const utterance = (ctx.currentTurn ?? '').trim()
      const committed: Array<{ level: string; id: string; label: string }> = []
      const corrected: Array<{ level: string; toId: string }> = []
      const needsSelection: Array<{ level: string; candidates: Array<{ id: string; label: string; path: string[] }> }> = []
      const notFound: string[] = []
      const messages: string[] = []

      // Process the next-unfilled level repeatedly; each decisive commit re-pins and
      // unlocks the following level within the SAME turn (handles "总部主楼网络部的王芳").
      for (;;) {
        const level = firstUnfilled(ctx)
        if (!level) break
        const pinned = pinnedFromWM(ctx)
        const lr = resolver.recall(utterance, pinned, recallConfig, level).byLevel[0]

        if (lr?.decisive) {
          const out = resolver.commit({ selected: lr.decisive, level, pinned })
          if (out.status === 'complete' || out.status === 'corrected') {
            ctx.workingMemory.set(level, out.resolved.id)
            committed.push({ level, id: out.resolved.id, label: out.resolved.label })
            if (out.status === 'corrected') {
              corrected.push({ level, toId: out.resolved.id })
              messages.push(repairMessages.noticeCorrected(resolver.levelLabel(level), out.resolved.label))
            }
            continue   // re-pinned → try the next level this same turn
          }
          // Decisive recall but commit rejected (conflicts with pinned branch) → stop.
          notFound.push(level)
          messages.push(repairMessages.promptForLevel(resolver.levelLabel(level)))
          break
        }

        if (lr && lr.candidates.length > 0) {
          needsSelection.push({
            level,
            candidates: lr.candidates.map(c => ({ id: c.id, label: c.label, path: c.path })),
          })
          messages.push(repairMessages.clarifyAmbiguous(resolver.levelLabel(level), lr.candidates))
          break   // first ambiguous level halts the deterministic run
        }

        notFound.push(level)
        messages.push(repairMessages.promptForLevel(resolver.levelLabel(level)))
        break
      }

      return { committed, corrected, needsSelection, notFound, message: messages.join(' ') || null }
    },
  }
}
