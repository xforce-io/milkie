// src/__tests__/AgentRuntime.previousRunId.test.ts
import { AgentRuntime } from '../runtime/AgentRuntime.js'
import { DefaultIOPort } from '../runtime/IOPort.js'
import { MemoryStore } from '../store/MemoryStore.js'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder.js'
import type { AgentConfig } from '../types/agent.js'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model.js'
import type { ToolDefinition } from '../types/tool.js'

function makeConfig(): AgentConfig {
  return {
    agentId: 'test-agent', version: '1.0.0', systemPrompt: 'test',
    fsm: { states: [{ name: 'react', type: 'llm' }] },
    model: { provider: 'test', model: 'test-model', adapter: 'test' },
  }
}
class SequentialGateway implements IModelGateway {
  private i = 0
  constructor(private responses: ModelResponse[]) {}
  async complete(_r: ModelRequest): Promise<ModelResponse> {
    const r = this.responses[this.i++]; if (!r) throw new Error('no more'); return r
  }
  async *stream(_r: ModelRequest): AsyncIterable<never> { yield* [] }
}

describe('#189 D1 ToolContext.previousRunId', () => {
  it('LLM-state tool handler receives ctx.previousRunId from runtime opts', async () => {
    let captured: string | undefined
    const probe: ToolDefinition = {
      name: 'probe', description: 'capture previousRunId',
      inputSchema: { type: 'object', properties: {} },
      handler: async (_input, ctx) => { captured = ctx?.previousRunId; return { ok: true } },
    }
    const gateway = new SequentialGateway([
      { content: [{ type: 'tool_use', id: 't1', name: 'probe', input: {} }],
        toolCalls: [{ id: 't1', name: 'probe', input: {} }], finishReason: 'tool_use' },
      { content: [{ type: 'text', text: 'done' }], toolCalls: [], finishReason: 'end_turn' },
    ])
    const runtime = new AgentRuntime({
      config: makeConfig(), goal: 'g', input: 'hi',
      previousRunId: 'run-prev-123',
      stateStore: new MemoryStore(),
      recorder: new InMemoryRecorder(),
      ioPort: new DefaultIOPort(gateway),
      extraTools: [probe],
    } as never)
    await runtime.run('hi')
    expect(captured).toBe('run-prev-123')
  })
})
