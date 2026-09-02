import { createHash } from 'node:crypto'

import type { Message } from '../types/common.js'
import type { ToolSchema } from '../types/model.js'

export const CONTEXT_BUDGET_REGIONS = [
  'control',
  'currentTurn',
  'scratchpad',
  'workingMemory',
  'sessionContext',
  'history',
  'externalProjection',
] as const

export type ContextBudgetRegion = typeof CONTEXT_BUDGET_REGIONS[number]
export type ContextBudgetTarget = 'system' | 'message' | 'tool'

export interface ContextBudgetConfig {
  /** Conservative upper bound for one model request. Defaults to 32768. */
  maxInputTokens?: number
  /** Optional per-region upper bounds. Omitted entries use DEFAULT_REGION_CAPS. */
  regionCaps?: Partial<Record<ContextBudgetRegion, number>>
}

export interface ContextBudgetItem {
  readonly id: string
  readonly region: ContextBudgetRegion
  readonly target: ContextBudgetTarget
  readonly content: string | Message[] | ToolSchema[]
  /** Assembly order, retained after budget projection. */
  readonly order: number
  /** Consecutive user-message fragments that must remain one message. */
  readonly mergeKey?: string
  /** Removed when this is the only surviving fragment of mergeKey. */
  readonly mergePrefix?: string
  /** Stable source handle for the retained raw Trace/checkpoint evidence. */
  readonly evidenceRef?: string
}

export interface ContextProjectionNotice {
  readonly region: ContextBudgetRegion
  readonly sourceId: string
  readonly originalBudgetTokens: number
  readonly projectedBudgetTokens: number
  readonly reason: 'region_cap' | 'total_cap'
  readonly contentHash: string
  readonly evidenceRef: string
}

export interface ContextBudgetReport {
  readonly totalLimit: number
  readonly totalEstimated: number
  readonly contextEpoch: number
  readonly regions: ReadonlyArray<{
    region: ContextBudgetRegion
    candidateEstimated: number
    projectedEstimated: number
  }>
  readonly notices: readonly ContextProjectionNotice[]
}

export interface BudgetedContext {
  readonly system: string
  readonly messages: Message[]
  readonly tools?: ToolSchema[]
  readonly report: ContextBudgetReport
}

export class ContextBudgetError extends Error {
  readonly retryable = false as const
  readonly phase = 'context_budget' as const

  constructor(
    readonly code: 'CONTEXT_BUDGET_INVALID_CONFIG' | 'CONTEXT_BUDGET_REQUIRED_REGION_EXCEEDED',
    readonly region: ContextBudgetRegion | undefined,
    readonly limit: number,
    readonly estimated: number,
  ) {
    super(code === 'CONTEXT_BUDGET_INVALID_CONFIG'
      ? 'Context budget configuration is invalid.'
      : 'Required context region exceeds the configured budget.')
    this.name = 'ContextBudgetError'
  }
}

export const DEFAULT_CONTEXT_BUDGET = 32768

export const DEFAULT_REGION_CAPS: Readonly<Record<ContextBudgetRegion, number>> = {
  control:            8192,
  currentTurn:        4096,
  scratchpad:         8192,
  workingMemory:      4096,
  sessionContext:     2048,
  history:            4096,
  externalProjection: 2048,
}

const REQUIRED_REGIONS = new Set<ContextBudgetRegion>(['control', 'currentTurn'])
const OPTIONAL_ORDER: readonly ContextBudgetRegion[] = [
  'scratchpad',
  'workingMemory',
  'sessionContext',
  'history',
  'externalProjection',
]

interface ResolvedContextBudget {
  readonly totalLimit: number
  readonly caps: Readonly<Record<ContextBudgetRegion, number>>
}

/**
 * Produce the exact model-visible projection for a region-backed assembled
 * context. The estimator is deliberately conservative: every UTF-8 byte is
 * charged as one input token, so it needs no provider tokenizer or locale.
 */
export function applyContextBudget(
  items: readonly ContextBudgetItem[],
  config: ContextBudgetConfig | undefined,
  contextEpoch: number,
): BudgetedContext {
  const budget = resolveContextBudget(config)
  const selected: ContextBudgetItem[] = []
  const notices: ContextProjectionNotice[] = []

  for (const region of CONTEXT_BUDGET_REGIONS) {
    if (!REQUIRED_REGIONS.has(region)) continue
    const candidates = items.filter(item => item.region === region)
    const candidateEstimate = estimateItems(candidates)
    if (candidateEstimate > budget.caps[region]) {
      throw requiredRegionError(region, budget.caps[region], candidateEstimate)
    }
    selected.push(...candidates)
  }

  if (estimateRequest(selected) > budget.totalLimit) {
    throw requiredRegionError(undefined, budget.totalLimit, estimateRequest(selected))
  }

  for (const region of OPTIONAL_ORDER) {
    const candidates = items.filter(item => item.region === region)
    const cap = budget.caps[region]
    if (candidates.length === 0 || cap === 0) continue

    const available = Math.min(cap, Math.max(0, budget.totalLimit - estimateRequest(selected)))
    if (available === 0) {
      notices.push(...candidates.map(item => notice(item, 0, 'total_cap')))
      continue
    }

    if (region === 'history' || region === 'externalProjection') {
      addNewestWholeItems(selected, candidates, available, budget.totalLimit, notices)
      continue
    }

    const projected = projectRegion(region, candidates, available, notices)
    for (const item of projected) {
      const withItem = [...selected, item]
      if (estimateRequest(withItem) <= budget.totalLimit) {
        selected.push(item)
      } else {
        notices.push(notice(item, 0, 'total_cap'))
      }
    }
  }

  const ordered = selected.slice().sort((a, b) => a.order - b.order)
  const finalEstimate = estimateRequest(ordered)
  if (finalEstimate > budget.totalLimit) {
    throw requiredRegionError(undefined, budget.totalLimit, finalEstimate)
  }

  const system = ordered
    .filter((item): item is ContextBudgetItem & { content: string } => item.target === 'system')
    .map(item => item.content)
    .join('\n')
  const messages = messagesFromBudgetItems(ordered)
  const tools = ordered
    .filter((item): item is ContextBudgetItem & { content: ToolSchema[] } => item.target === 'tool')
    .flatMap(item => item.content)

  return {
    system,
    messages,
    ...(tools.length > 0 ? { tools } : {}),
    report: {
      totalLimit: budget.totalLimit,
      totalEstimated: finalEstimate,
      contextEpoch,
      regions: CONTEXT_BUDGET_REGIONS.map(region => ({
        region,
        candidateEstimated: estimateItems(items.filter(item => item.region === region)),
        projectedEstimated: estimateItems(ordered.filter(item => item.region === region)),
      })),
      notices,
    },
  }
}

export function estimateBudgetTokens(value: unknown): number {
  return Buffer.byteLength(JSON.stringify(value), 'utf8')
}

function resolveContextBudget(config: ContextBudgetConfig | undefined): ResolvedContextBudget {
  const totalLimit = config?.maxInputTokens ?? DEFAULT_CONTEXT_BUDGET
  if (!isPositiveInteger(totalLimit)) {
    throw new ContextBudgetError('CONTEXT_BUDGET_INVALID_CONFIG', undefined, DEFAULT_CONTEXT_BUDGET, Number(totalLimit) || 0)
  }

  const caps: Record<ContextBudgetRegion, number> = { ...DEFAULT_REGION_CAPS }
  for (const [region, cap] of Object.entries(config?.regionCaps ?? {})) {
    if (!CONTEXT_BUDGET_REGIONS.includes(region as ContextBudgetRegion) || !isPositiveInteger(cap) || cap > totalLimit) {
      throw new ContextBudgetError('CONTEXT_BUDGET_INVALID_CONFIG', region as ContextBudgetRegion, totalLimit, Number(cap) || 0)
    }
    caps[region as ContextBudgetRegion] = cap
  }
  return { totalLimit, caps }
}

function addNewestWholeItems(
  selected: ContextBudgetItem[],
  candidates: readonly ContextBudgetItem[],
  cap: number,
  totalLimit: number,
  notices: ContextProjectionNotice[],
): void {
  let used = 0
  const kept = new Set<string>()
  for (const item of candidates.slice().reverse()) {
    const itemEstimate = estimateBudgetTokens(item.content)
    if (used + itemEstimate <= cap && estimateRequest([...selected, item]) <= totalLimit) {
      selected.push(item)
      used += itemEstimate
      kept.add(item.id)
    }
  }
  for (const item of candidates) {
    if (!kept.has(item.id)) notices.push(notice(item, 0, used >= cap ? 'region_cap' : 'total_cap'))
  }
}

function projectRegion(
  region: ContextBudgetRegion,
  candidates: readonly ContextBudgetItem[],
  cap: number,
  notices: ContextProjectionNotice[],
): ContextBudgetItem[] {
  const candidateEstimate = estimateItems(candidates)
  if (candidateEstimate <= cap) return [...candidates]

  if (region === 'scratchpad') {
    const projected: ContextBudgetItem[] = []
    let remaining = cap
    for (const original of candidates) {
      const replacement = shrinkScratchpad(original, remaining)
      const projectedEstimate = estimateBudgetTokens(replacement.content)
      if (projectedEstimate > remaining) {
        throw requiredRegionError('scratchpad', cap, candidateEstimate)
      }
      if (estimateBudgetTokens(replacement.content) < estimateBudgetTokens(original.content)) {
        notices.push(notice(original, estimateBudgetTokens(replacement.content), 'region_cap'))
      }
      projected.push(replacement)
      remaining -= projectedEstimate
    }
    return projected
  }

  if (region === 'workingMemory') {
    const first = candidates[0]
    if (!first || first.target !== 'system' || typeof first.content !== 'string') return []
    const replacement = { ...first, content: truncateText(first.content, cap, first.id) }
    notices.push(notice(first, estimateBudgetTokens(replacement.content), 'region_cap'))
    return [replacement]
  }

  // Session context is a collection of independently rendered variables. Keep
  // deterministic prefix items and explicitly report every omitted source.
  const kept: ContextBudgetItem[] = []
  let used = 0
  for (const item of candidates) {
    const size = estimateBudgetTokens(item.content)
    if (used + size <= cap) {
      kept.push(item)
      used += size
    } else {
      notices.push(notice(item, 0, 'region_cap'))
    }
  }
  return kept
}

function shrinkScratchpad(item: ContextBudgetItem, cap: number): ContextBudgetItem {
  if (item.target !== 'message' || !Array.isArray(item.content)) return item
  const originalMessages = item.content as Message[]
  if (estimateBudgetTokens(originalMessages) <= cap) return item

  const messages: Message[] = originalMessages.map(message => ({ ...message, content: message.content.map(part => ({ ...part })) }))
  const results: Array<{ messageIndex: number; contentIndex: number }> = []
  for (let messageIndex = 0; messageIndex < messages.length; messageIndex++) {
    const message = messages[messageIndex]!
    for (let contentIndex = 0; contentIndex < message.content.length; contentIndex++) {
      if (message.content[contentIndex]?.type === 'tool_result') results.push({ messageIndex, contentIndex })
    }
  }
  if (results.length === 0) return item

  const empty = messages.map(message => ({
    ...message,
    content: message.content.map(part => part.type === 'tool_result' ? { ...part, content: '' } : part),
  }))
  const reserved = estimateBudgetTokens(empty)
  if (reserved > cap) return item

  const perResult = Math.max(0, Math.floor((cap - reserved) / results.length))
  for (const { messageIndex, contentIndex } of results) {
    const part = messages[messageIndex]!.content[contentIndex]!
    if (part.type === 'tool_result') {
      messages[messageIndex]!.content[contentIndex] = {
        ...part,
        content: truncateText(part.content, perResult, `scratch:${part.tool_use_id}`),
      }
    }
  }
  return { ...item, content: messages }
}

function truncateText(value: string, maxBudgetTokens: number, sourceId: string): string {
  if (estimateBudgetTokens(value) <= maxBudgetTokens) return value
  const marker = `\n[context-budget omitted; source=${sourceId}; hash=${contentHash(value)}]`
  if (estimateBudgetTokens(marker) >= maxBudgetTokens) return marker.slice(0, Math.max(0, maxBudgetTokens))

  const chars = Array.from(value)
  let low = 0
  let high = chars.length
  while (low < high) {
    const middle = Math.ceil((low + high) / 2)
    if (estimateBudgetTokens(chars.slice(0, middle).join('') + marker) <= maxBudgetTokens) low = middle
    else high = middle - 1
  }
  return chars.slice(0, low).join('') + marker
}

function estimateItems(items: readonly ContextBudgetItem[]): number {
  return items.reduce((total, item) => total + estimateBudgetTokens(item.content), 0)
}

function estimateRequest(items: readonly ContextBudgetItem[]): number {
  const ordered = items.slice().sort((a, b) => a.order - b.order)
  const system = ordered.filter(item => item.target === 'system').map(item => item.content).join('\n')
  const messages = messagesFromBudgetItems(ordered)
  const tools = ordered.filter(item => item.target === 'tool').flatMap(item => item.content as ToolSchema[])
  return estimateBudgetTokens({ system, messages, ...(tools.length > 0 ? { tools } : {}) })
}

export function messagesFromBudgetItems(items: readonly ContextBudgetItem[]): Message[] {
  const messages: Message[] = []
  let previousMergeKey: string | undefined
  for (const item of items) {
    if (item.target !== 'message') continue
    const rendered = (item.content as Message[]).map(message => ({
      ...message,
      content: message.content.map(part => ({ ...part })),
    }))
    if (item.mergeKey && rendered.length === 1 && rendered[0]?.role === 'user') {
      const text = rendered[0].content.length === 1 && rendered[0].content[0]?.type === 'text'
        ? rendered[0].content[0].text
        : undefined
      const previous = messages[messages.length - 1]
      if (previousMergeKey === item.mergeKey && previous?.role === 'user' && text !== undefined) {
        const previousText = previous.content.length === 1 && previous.content[0]?.type === 'text'
          ? previous.content[0].text
          : undefined
        if (previousText !== undefined) {
          previous.content = [{ type: 'text', text: `${previousText}\n\n${text}` }]
          continue
        }
      }
      if (text !== undefined && item.mergePrefix && text.startsWith(item.mergePrefix)) {
        messages.push({ role: 'user', content: [{ type: 'text', text: text.slice(item.mergePrefix.length) }] })
        previousMergeKey = item.mergeKey
        continue
      }
    }
    messages.push(...rendered)
    previousMergeKey = item.mergeKey
  }
  return messages
}

function notice(
  item: ContextBudgetItem,
  projectedBudgetTokens: number,
  reason: ContextProjectionNotice['reason'],
): ContextProjectionNotice {
  return {
    region: item.region,
    sourceId: item.id,
    originalBudgetTokens: estimateBudgetTokens(item.content),
    projectedBudgetTokens,
    reason,
    contentHash: contentHash(item.content),
    evidenceRef: item.evidenceRef ?? item.id,
  }
}

function contentHash(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex')
}

function requiredRegionError(region: ContextBudgetRegion | undefined, limit: number, estimated: number): ContextBudgetError {
  return new ContextBudgetError('CONTEXT_BUDGET_REQUIRED_REGION_EXCEEDED', region, limit, estimated)
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0
}
