// #128: Milkie.getSessionHistory(contextId) — the full, per-message transcript
// of a whole session (every run/turn under one contextId), with tool chains
// intact, projected from each run's event log and concatenated in turn order.
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'
import type { ToolDefinition } from '../types/tool'
import type { Message, MessageContent } from '../types/common'

const recorderAgent: AgentConfig = {
  agentId: 'recorder', version: '1.0.0', systemPrompt: 'answer',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: 50 }] },
  model: { provider: 'test', model: 'test', adapter: 'test' },
}

function factWriter(): ToolDefinition {
  return {
    name: 'record_fact', description: 'record a fact',
    inputSchema: { type: 'object', properties: { key: { type: 'string' }, value: { type: 'string' } }, required: ['key', 'value'] },
    handler: async (input: unknown, ctx) => {
      const { key, value } = input as { key: string; value: string }
      ctx.workingMemory.set(key, value)
      return { recorded: key }
    },
  }
}

/** Each turn: first call records fact{n} (tool_use), second call completes. */
function multiTurnRecorder(): IModelGateway {
  let calls = 0, facts = 0
  return {
    async complete(_req: ModelRequest): Promise<ModelResponse> {
      calls++
      if (calls % 2 === 1) {
        facts++
        const input = { key: `fact${facts}`, value: `v${facts}` }
        return { content: [{ type: 'tool_use', id: `c${facts}`, name: 'record_fact', input }], toolCalls: [{ id: `c${facts}`, name: 'record_fact', input }], finishReason: 'tool_use' }
      }
      return { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'end_turn' }
    },
    async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] },
  }
}

const userTexts = (msgs: Message[]): string[] =>
  msgs.filter(m => m.role === 'user').map(m => (m.content[0] as { type: 'text'; text: string }).text)

describe('#128 Milkie.getSessionHistory', () => {
  it('returns the full per-turn transcript across runs, tool chains intact, no duplication', async () => {
    const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore(), gateway: multiTurnRecorder(), tools: [factWriter()] })
    milkie.registerAgent(recorderAgent)
    const contextId = 'ctx-hist'

    await milkie.invoke({ agentId: 'recorder', goal: 'g', input: 'turn1', contextId })  // run A → fact1
    await milkie.invoke({ agentId: 'recorder', goal: 'g', input: 'turn2', contextId })  // run B → fact2

    const messages = await milkie.getSessionHistory(contextId)

    // Both turns present, in order, each exactly once (no restored-prefix dup).
    expect(userTexts(messages)).toEqual(['turn1', 'turn2'])

    // Per turn: user, assistant(tool_use), tool_result, assistant(final) = 4 × 2.
    expect(messages).toHaveLength(8)

    // Turn 1's tool chain is paired and complete.
    const toolUse1 = messages[1]!.content[0] as Extract<MessageContent, { type: 'tool_use' }>
    expect(messages[1]!.role).toBe('assistant')
    expect(toolUse1).toMatchObject({ type: 'tool_use', id: 'c1', name: 'record_fact', input: { key: 'fact1' } })
    expect(messages[2]!).toEqual({ role: 'tool', content: [{ type: 'tool_result', tool_use_id: 'c1', content: '{"recorded":"fact1"}' }] })
    expect(messages[3]!).toEqual({ role: 'assistant', content: [{ type: 'text', text: 'done' }] })

    // Turn 2's tool_use pairs with turn 2's tool_result (fact2 / c2), not turn 1's.
    const toolUse2 = messages[5]!.content[0] as Extract<MessageContent, { type: 'tool_use' }>
    expect(toolUse2).toMatchObject({ type: 'tool_use', id: 'c2', name: 'record_fact', input: { key: 'fact2' } })
    expect(messages[6]!).toEqual({ role: 'tool', content: [{ type: 'tool_result', tool_use_id: 'c2', content: '{"recorded":"fact2"}' }] })
  })

  it('throws for an unknown contextId (no session)', async () => {
    const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: new MemoryEventStore(), gateway: multiTurnRecorder(), tools: [factWriter()] })
    milkie.registerAgent(recorderAgent)
    await expect(milkie.getSessionHistory('never')).rejects.toThrow(/never/)
  })
})
