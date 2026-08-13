import type { ArtifactRef, DeliverableSpec, StopReason } from '../types/common.js'

export class DeliverableDeclarationError extends Error {
  readonly code = 'DELIVERABLE_INVALID' as const
  constructor(message: string) {
    super(message)
    this.name = 'DeliverableDeclarationError'
  }
}

export function normalizePath(p: string): string {
  const trimmed = p.trim()
  const parts = trimmed.replace(/\\/g, '/').split('/')
  const out: string[] = []
  for (const part of parts) {
    if (part === '' || part === '.') continue
    if (part === '..') {
      throw new DeliverableDeclarationError(`deliverable path must not escape the work root: "${p}"`)
    }
    out.push(part)
  }
  return out.join('/')
}

export function validateDeliverableList(list: readonly DeliverableSpec[]): DeliverableSpec[] {
  const seen = new Set<string>()
  const out: DeliverableSpec[] = []
  for (const item of list) {
    if (!item || typeof item.name !== 'string' || item.name.trim() === '') {
      throw new DeliverableDeclarationError('each deliverable requires a non-empty name')
    }
    const name = item.name.trim()
    if (seen.has(name)) {
      throw new DeliverableDeclarationError(`duplicate deliverable name "${name}"`)
    }
    seen.add(name)
    if (item.type !== 'file' && item.type !== 'object') {
      throw new DeliverableDeclarationError(`deliverable "${name}" has invalid type`)
    }
    if (item.type === 'file') {
      if (!item.path || typeof item.path !== 'string' || item.path.trim() === '') {
        throw new DeliverableDeclarationError(`file deliverable "${name}" requires path`)
      }
      out.push({
        name,
        type: 'file',
        path: normalizePath(item.path),
        required: item.required !== false,
      })
    } else {
      out.push({
        name,
        type: 'object',
        required: item.required !== false,
      })
    }
  }
  return out
}

/**
 * #247: invoke key present (including []) replaces agent default wholesale.
 * Both omitted → no contract (`null`).
 */
export function resolveEffectiveDeliverables(
  agentList: readonly DeliverableSpec[] | undefined,
  invokeSpecified: boolean,
  invokeList: readonly DeliverableSpec[] | undefined,
): DeliverableSpec[] | null {
  if (invokeSpecified) return validateDeliverableList(invokeList ?? [])
  if (agentList !== undefined) return validateDeliverableList(agentList)
  return null
}

export interface ProducedRecord {
  type: ArtifactTypeLike
  path?: string
  objectId?: string
  name?: string
  hash?: string
}

type ArtifactTypeLike = 'file' | 'object'

export function matchArtifacts(
  contract: readonly DeliverableSpec[] | null,
  produced: readonly ProducedRecord[],
): ArtifactRef[] {
  if (contract === null) {
    return produced
      .filter(p => p.path || p.objectId || p.name)
      .map(p => ({
        name: p.name ?? p.path ?? p.objectId ?? 'artifact',
        type: p.type,
        ...(p.path ? { path: p.path } : {}),
        ...(p.objectId ? { objectId: p.objectId } : {}),
        state: 'produced' as const,
        ...(p.hash ? { hash: p.hash } : {}),
      }))
  }

  return contract.map(spec => {
    const hit = produced.find(p => matchesSpec(spec, p))
    if (!hit) {
      return {
        name: spec.name,
        type: spec.type,
        ...(spec.path ? { path: spec.path } : {}),
        state: 'missing' as const,
      }
    }
    return {
      name: spec.name,
      type: spec.type,
      ...(hit.path ? { path: hit.path } : spec.path ? { path: spec.path } : {}),
      ...(hit.objectId ? { objectId: hit.objectId } : {}),
      state: 'produced' as const,
      ...(hit.hash ? { hash: hit.hash } : {}),
    }
  })
}

function matchesSpec(spec: DeliverableSpec, produced: ProducedRecord): boolean {
  if (spec.type === 'file') {
    if (produced.type !== 'file' || !spec.path || !produced.path) return false
    return normalizePath(produced.path) === spec.path
  }
  if (produced.type !== 'object') return false
  if (produced.name && produced.name === spec.name) return true
  return false
}

export function partialFromContract(artifacts: readonly ArtifactRef[]): boolean {
  return artifacts.some(a => a.state === 'missing' && isRequiredName(a, artifacts))
}

function isRequiredName(artifact: ArtifactRef, _all: readonly ArtifactRef[]): boolean {
  return artifact.state === 'missing'
}

export function computePartial(opts: {
  contract: readonly DeliverableSpec[] | null
  artifacts: readonly ArtifactRef[]
  stopReason: StopReason
}): boolean {
  if (opts.contract !== null) {
    const requiredMissing = opts.contract.some(spec => {
      if (spec.required === false) return false
      const row = opts.artifacts.find(a => a.name === spec.name)
      return !row || row.state === 'missing'
    })
    return requiredMissing
  }
  return opts.stopReason !== 'model_stop'
}
