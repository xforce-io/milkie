import type { ToolDefinition, ToolContext } from '../types/tool.js'
import type { RelationType } from '../trace/types.js'

/**
 * #113 P3: framework built-in lineage-declaration tools.
 *
 * Object *production* (a passage read, a row fetched) belongs to the data tools
 * (e.g. the corpus tools' read_file/grep), which mint objects via
 * ctx.createObject / ctx.registerObject. These tools only *declare relations*
 * into the lineage graph — they are the agent-facing entry point for the
 * lineage capability, available to any agent (gated by each state's `tools`
 * allowlist). They lean on the runtime primitives added in P1/P2:
 *   - resolveObject  → fail-fast on a fabricated/hallucinated objectId
 *   - promoteObject  → emit a lazily-registered object on first citation
 *   - createObject / createRelation → record claim + typed edge
 */

function unresolved(ctx: ToolContext, objectId: string): boolean {
  return !!ctx.resolveObject && !ctx.resolveObject(objectId)
}

async function citeHandler(input: unknown, ctx: ToolContext): Promise<unknown> {
  const { claim, objectId } = input as { claim: string; objectId: string }
  if (unresolved(ctx, objectId)) {
    return { ok: false, error: `objectId '${objectId}' 不存在；请使用工具返回的真实 objectId` }
  }
  ctx.promoteObject?.(objectId)
  const claimObj = ctx.createObject?.({ type: 'claim', meta: { text: claim } })
  if (claimObj && ctx.createRelation) {
    ctx.createRelation({ type: 'cites', fromObjectId: claimObj.objectId, toObjectId: objectId })
  }
  return { ok: true, claimId: claimObj?.objectId, cites: objectId }
}

async function declareRelationHandler(input: unknown, ctx: ToolContext): Promise<unknown> {
  const { type, fromObjectId, toObjectId } = input as { type: RelationType; fromObjectId: string; toObjectId: string }
  for (const id of [fromObjectId, toObjectId]) {
    if (unresolved(ctx, id)) {
      return { ok: false, error: `objectId '${id}' 不存在；请使用工具返回的真实 objectId` }
    }
  }
  ctx.promoteObject?.(fromObjectId)
  ctx.promoteObject?.(toObjectId)
  ctx.createRelation?.({ type, fromObjectId, toObjectId })
  return { ok: true, type, fromObjectId, toObjectId }
}

export const lineageTools: ToolDefinition[] = [
  {
    name:        'cite',
    description: 'Record that a claim in your answer is sourced from (cites) an object. Pass the exact claim text and an objectId returned by a data tool (e.g. read_file/grep). Call once per cited claim; never write "(chapter:line)" in prose. Returns { ok:false, error } if the objectId is not a real one you received — re-fetch and retry.',
    inputSchema: {
      type: 'object',
      properties: {
        claim:    { type: 'string', description: 'The exact statement this source supports.' },
        objectId: { type: 'string', description: 'objectId of the supporting object (from a data tool).' },
      },
      required: ['claim', 'objectId'],
    },
    handler: citeHandler,
  },
  {
    name:        'declare_relation',
    description: 'Declare a typed lineage edge between two existing objects (cites / derives_from / supersedes / equivalent_to). Both objectIds must be ones returned by data tools. Returns { ok:false, error } for an unknown objectId.',
    inputSchema: {
      type: 'object',
      properties: {
        type:         { type: 'string', enum: ['cites', 'derives_from', 'supersedes', 'equivalent_to'], description: 'Relation type.' },
        fromObjectId: { type: 'string', description: 'Source objectId.' },
        toObjectId:   { type: 'string', description: 'Target objectId.' },
      },
      required: ['type', 'fromObjectId', 'toObjectId'],
    },
    handler: declareRelationHandler,
  },
]
