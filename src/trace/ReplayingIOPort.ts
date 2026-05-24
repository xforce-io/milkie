import type { IIOPort } from '../runtime/IOPort.js'
import type { ModelRequest, ModelResponse } from '../types/model.js'
import type { CacheIndex } from './CacheIndex.js'
import { hashModelRequest, hashToolCall } from './hash.js'
import { ReplayDivergenceError } from './ReplayDivergenceError.js'

/**
 * IOPort implementation that serves LLM/tool calls from a CacheIndex
 * built from a recorded run's events. Cache miss → ReplayDivergenceError.
 * inner is used only for now()/uuid() passthrough — its invokeLLM /
 * invokeTool are never called during replay.
 */
export class ReplayingIOPort implements IIOPort {
  constructor(
    private readonly cache: CacheIndex,
    private readonly inner: IIOPort,
  ) {}

  async invokeLLM(request: ModelRequest): Promise<ModelResponse> {
    const hash = hashModelRequest(request)
    try {
      return this.cache.consumeLLM(hash)
    } catch {
      const lastUserMessage = request.messages
        .filter(m => m.role === 'user')
        .flatMap(m => m.content)
        .map(c => c.type === 'text' ? c.text : `[${c.type}]`)
        .pop() ?? ''
      const summary = `model=${request.model} lastUser=${lastUserMessage.slice(0, 80)}`
      throw new ReplayDivergenceError('llm', hash, summary, this.cache.allHashes().llm.slice(0, 5))
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
      // consumeTool throws a normal Error for "queue empty"; rethrows reconstructed
      // tool errors for recorded failures. Distinguish by message prefix.
      if (err instanceof Error && err.message.startsWith('CacheIndex: tool queue empty')) {
        const summary = `toolName=${toolName} input=${JSON.stringify(input).slice(0, 80)}`
        throw new ReplayDivergenceError('tool', hash, summary, this.cache.allHashes().tool.slice(0, 5))
      }
      throw err
    }
  }

  now(): number {
    return this.inner.now()
  }

  uuid(): string {
    return this.inner.uuid()
  }
}
