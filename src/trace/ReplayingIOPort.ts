import type { IIOPort } from '../runtime/IOPort.js'
import type { ModelRequest, ModelResponse, ModelEvent } from '../types/model.js'
import { CacheIndex, CacheIndexEmptyError } from './CacheIndex.js'
import { hashModelRequest, hashToolCall } from './hash.js'
import { ReplayDivergenceError } from './ReplayDivergenceError.js'

/**
 * IOPort implementation that serves every IIOPort method from a CacheIndex
 * built from a recorded run's events: LLM/tool calls keyed by request hash,
 * clock/uuid by FIFO position. Cache miss → ReplayDivergenceError.
 * `inner` is retained for type contract symmetry with RecordingIOPort but is
 * never called during replay; touching it from this class is a bug.
 */
export class ReplayingIOPort implements IIOPort {
  constructor(
    private readonly cache: CacheIndex,
    private readonly inner: IIOPort,
  ) {}

  async invokeLLM(request: ModelRequest, _onEvent?: (e: ModelEvent) => void): Promise<ModelResponse> {
    const hash = hashModelRequest(request)
    try {
      return this.cache.consumeLLM(hash)
    } catch (err) {
      if (err instanceof CacheIndexEmptyError) {
        const lastUserMessage = request.messages
          .filter(m => m.role === 'user')
          .flatMap(m => m.content)
          .map(c => c.type === 'text' ? c.text : `[${c.type}]`)
          .pop() ?? ''
        const summary = `model=${request.model} lastUser=${lastUserMessage.slice(0, 80)}`
        throw new ReplayDivergenceError('llm', hash, summary, this.cache.allHashes().llm.slice(0, 5))
      }
      throw err
    }
  }

  async invokeTool(
    toolName: string,
    input: unknown,
    _execute: () => Promise<unknown>,
  ): Promise<unknown> {
    const hash = hashToolCall(toolName, input)
    try {
      return this.cache.consumeTool(hash)
    } catch (err) {
      // consumeTool throws CacheIndexEmptyError for "queue empty"; rethrows
      // reconstructed tool errors for recorded failures. Distinguish structurally.
      if (err instanceof CacheIndexEmptyError) {
        const summary = `toolName=${toolName} input=${JSON.stringify(input).slice(0, 80)}`
        throw new ReplayDivergenceError('tool', hash, summary, this.cache.allHashes().tool.slice(0, 5))
      }
      throw err
    }
  }

  now(): number {
    try {
      return this.cache.consumeClock()
    } catch (err) {
      if (err instanceof CacheIndexEmptyError) {
        const r = this.cache.remaining()
        throw new ReplayDivergenceError(
          'clock', '',
          `clock.read queue exhausted (remaining llm=${r.llm} tool=${r.tool} uuid=${r.uuid})`,
          [],
        )
      }
      throw err
    }
  }

  uuid(): string {
    try {
      return this.cache.consumeUuid()
    } catch (err) {
      if (err instanceof CacheIndexEmptyError) {
        const r = this.cache.remaining()
        throw new ReplayDivergenceError(
          'uuid', '',
          `uuid.generated queue exhausted (remaining llm=${r.llm} tool=${r.tool} clock=${r.clock})`,
          [],
        )
      }
      throw err
    }
  }
}
