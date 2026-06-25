/**
 * #200 C: AgentRuntime exposes ctx.deliveredRunIds — the sourceRunIds of the
 * external projections delivered to this run — so the selfOnly trace tools can
 * gate runId dereference on "was this delivered to me".
 */
import { AgentRuntime } from '../runtime/AgentRuntime'
import { DefaultIOPort } from '../runtime/IOPort'
import { MemoryStore } from '../store/MemoryStore'
import { InMemoryRecorder } from '../trajectory/InMemoryRecorder'
import type { AgentConfig } from '../types/agent'
import type { ContextProjection } from '../types/common'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { ToolDefinition } from '../types/tool'

function makeConfig(): AgentConfig {
  return {
    agentId:      'test-agent',
    version:      '1.0.0',
    systemPrompt: 'You are a test agent.',
    fsm:          { states: [{ name: 'react', type: 'llm' }] },
    model:        { provider: 'test', model: 'test-model', adapter: 'test' },
  }
}

class SequentialGateway implements IModelGateway {
  private index = 0
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.responses[this.index++]
    if (!r) throw new Error('No more mock responses')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}

const toolCall = (id: string, name: string): ModelResponse => ({
  content: [{ type: 'tool_use', id, name, input: {} }], toolCalls: [{ id, name, input: {} }], finishReason: 'tool_use',
})
const text = (t: string): ModelResponse => ({ content: [{ type: 'text', text: t }], toolCalls: [], finishReason: 'end_turn' })

function projection(sourceRunId: string): ContextProjection {
  return { sourceRunId, displayText: 'a delivered report', deliveredAt: 1, attachedAt: 1 }
}

describe('#200 ToolContext.deliveredRunIds', () => {
  it('a tool handler sees the sourceRunIds of the delivered projections', async () => {
    let captured: string[] | undefined
    const probe: ToolDefinition = {
      name: 'probe', description: 'capture deliveredRunIds',
      inputSchema: { type: 'object', properties: {} },
      handler: async (_i, ctx) => { captured = ctx.deliveredRunIds; return { ok: true } },
    }
    const runtime = new AgentRuntime({
      config:     makeConfig(),
      goal:       'g',
      input:      'hi',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(new SequentialGateway([toolCall('t1', 'probe'), text('done')])),
      extraTools: [probe],
      externalProjections: [projection('job-run-1'), projection('job-run-2')],
    })
    await runtime.run('hi')
    expect(captured).toEqual(['job-run-1', 'job-run-2'])
  })

  it('with no projections, deliveredRunIds is empty/undefined (no axis opened)', async () => {
    let captured: string[] | undefined = ['sentinel']
    const probe: ToolDefinition = {
      name: 'probe', description: 'capture deliveredRunIds',
      inputSchema: { type: 'object', properties: {} },
      handler: async (_i, ctx) => { captured = ctx.deliveredRunIds; return { ok: true } },
    }
    const runtime = new AgentRuntime({
      config:     makeConfig(),
      goal:       'g',
      input:      'hi',
      stateStore: new MemoryStore(),
      recorder:   new InMemoryRecorder(),
      ioPort:     new DefaultIOPort(new SequentialGateway([toolCall('t1', 'probe'), text('done')])),
      extraTools: [probe],
    })
    await runtime.run('hi')
    expect(captured ?? []).toEqual([])
  })
})
