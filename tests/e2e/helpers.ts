import { EventEmitter } from 'events'
import type { ToolDefinition } from '../../src/types/tool.js'
import type { Trajectory } from '../../src/types/trajectory.js'

export const SKIP = !process.env['VOLCENGINE_TOKEN'] || !process.env['VOLCENGINE_API_BASE']

export const MODEL_CONFIG = {
  provider: 'volcengine',
  model:    'doubao-seed-2.0-lite',
  adapter:  'openai-compatible' as const,
}

/** Create a tool-call counter that emits events when tools are called */
export function createToolCallTracker() {
  const emitter = new EventEmitter()
  const counts: Record<string, number> = {}

  function track(toolName: string): void {
    counts[toolName] = (counts[toolName] ?? 0) + 1
    emitter.emit(toolName, counts[toolName])
    emitter.emit('*', toolName, counts[toolName])
  }

  function waitFor(toolName: string, count: number): Promise<void> {
    if ((counts[toolName] ?? 0) >= count) return Promise.resolve()
    return new Promise(resolve => {
      const listener = (n: number) => {
        if (n >= count) {
          emitter.removeListener(toolName, listener)
          resolve()
        }
      }
      emitter.on(toolName, listener)
    })
  }

  return { track, waitFor, counts }
}

/** Assert that spans for a given tool all have the same turn number */
export function assertSameTurn(spans: Trajectory['spans'], toolName: string): void {
  const toolSpans = spans.filter(s => s.name === 'tool.call' && s.attributes['toolName'] === toolName)
  if (toolSpans.length === 0) return
  const turns = toolSpans.map(s => s.attributes['turn'] as number)
  const unique = new Set(turns)
  if (unique.size !== 1) {
    throw new Error(`Expected all ${toolName} calls to be in same turn, got turns: ${[...unique].join(', ')}`)
  }
}

/** Get span duration (ms) */
export function spanDuration(span: Trajectory['spans'][number]): number {
  return (span.endTime ?? span.startTime) - span.startTime
}

/** Compute manifest diff (returns changed keys with [before, after] tuples) */
export function diffManifests(
  a: Record<string, unknown>,
  b: Record<string, unknown>,
): Record<string, [unknown, unknown]> {
  const diff: Record<string, [unknown, unknown]> = {}
  const keys = new Set([...Object.keys(a), ...Object.keys(b)])
  for (const k of keys) {
    if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) {
      diff[k] = [a[k], b[k]]
    }
  }
  return diff
}

export interface Experiment {
  id: string
  goal: string
  variants: Array<{
    name: string
    agentVersion: string
    trajectoryIds: string[]
  }>
}

/** Ensure test output directory exists */
import fs from 'fs'

export function ensureDir(dir: string): void {
  fs.mkdirSync(dir, { recursive: true })
}

// Satisfy unused-import check — ToolDefinition is used by callers via re-export
export type { ToolDefinition }
