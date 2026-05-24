import type {
  Event,
  LlmRespondedPayload,
  ToolRespondedPayload,
  ClockReadPayload,
  UuidGeneratedPayload,
} from './types.js'
import type { ModelResponse } from '../types/model.js'

/**
 * In-memory projection of LLM/tool response events keyed by canonical
 * request hash, with one FIFO queue per hash. Drives strict structural
 * replay: consume calls dequeue; replay throws when a queue empties
 * or the hash was never recorded.
 */
export class CacheIndex {
  private readonly llm:   Map<string, ModelResponse[]>
  private readonly tool:  Map<string, ToolOutcome[]>
  private readonly clock: number[]
  private readonly uuid:  string[]

  private constructor(
    llm:   Map<string, ModelResponse[]>,
    tool:  Map<string, ToolOutcome[]>,
    clock: number[],
    uuid:  string[],
  ) {
    this.llm   = llm
    this.tool  = tool
    this.clock = clock
    this.uuid  = uuid
  }

  static fromEvents(events: Event[]): CacheIndex {
    const llm:   Map<string, ModelResponse[]> = new Map()
    const tool:  Map<string, ToolOutcome[]>   = new Map()
    const clock: number[] = []
    const uuid:  string[] = []

    for (const ev of events) {
      if (ev.type === 'llm.responded') {
        const p = ev.payload as LlmRespondedPayload
        if (!p.requestHash) continue   // Phase 2 events; skip
        push(llm, p.requestHash, p.response)
      } else if (ev.type === 'tool.responded') {
        const p = ev.payload as ToolRespondedPayload
        if (!p.requestHash) continue   // Phase 2 events; skip
        push(tool, p.requestHash, { output: p.output, error: p.error })
      } else if (ev.type === 'clock.read') {
        clock.push((ev.payload as ClockReadPayload).value)
      } else if (ev.type === 'uuid.generated') {
        uuid.push((ev.payload as UuidGeneratedPayload).value)
      }
    }

    return new CacheIndex(llm, tool, clock, uuid)
  }

  consumeLLM(hash: string): ModelResponse {
    const q = this.llm.get(hash)
    if (!q || q.length === 0) throw new CacheIndexEmptyError(`CacheIndex: LLM queue empty for hash ${hash}`)
    return q.shift()!
  }

  consumeTool(hash: string): unknown {
    const q = this.tool.get(hash)
    if (!q || q.length === 0) throw new CacheIndexEmptyError(`CacheIndex: tool queue empty for hash ${hash}`)
    const outcome = q.shift()!
    if (outcome.error) {
      const err = new Error(outcome.error.message) as Error & { retryable?: boolean; code?: string }
      if (outcome.error.retryable !== undefined) err.retryable = outcome.error.retryable
      if (outcome.error.code !== undefined)      err.code      = outcome.error.code
      if (outcome.error.name !== undefined)      err.name      = outcome.error.name
      throw err
    }
    return outcome.output
  }

  consumeClock(): number {
    if (this.clock.length === 0) throw new CacheIndexEmptyError('CacheIndex: clock queue empty')
    return this.clock.shift()!
  }

  consumeUuid(): string {
    if (this.uuid.length === 0) throw new CacheIndexEmptyError('CacheIndex: uuid queue empty')
    return this.uuid.shift()!
  }

  remaining(): { llm: number; tool: number; clock: number; uuid: number } {
    let llmCount = 0, toolCount = 0
    for (const q of this.llm.values())  llmCount  += q.length
    for (const q of this.tool.values()) toolCount += q.length
    return { llm: llmCount, tool: toolCount, clock: this.clock.length, uuid: this.uuid.length }
  }

  allHashes(): { llm: string[]; tool: string[] } {
    return { llm: [...this.llm.keys()], tool: [...this.tool.keys()] }
  }
}

interface ToolOutcome {
  output?: unknown
  error?:  ToolRespondedPayload['error']
}

function push<K, V>(map: Map<K, V[]>, key: K, value: V): void {
  const q = map.get(key)
  if (q) q.push(value)
  else   map.set(key, [value])
}

/**
 * Thrown by CacheIndex.consumeLLM / consumeTool when the FIFO queue for a
 * given hash is empty (i.e. the replay has consumed all recorded responses).
 * Using a named class lets callers distinguish "queue exhausted" from a
 * reconstructed tool error (which is also an Error) without fragile
 * message-prefix matching.
 */
export class CacheIndexEmptyError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CacheIndexEmptyError'
  }
}
