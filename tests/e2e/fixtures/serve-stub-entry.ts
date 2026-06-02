/**
 * #86 e2e fixture: a `milkie serve` instance backed by a deterministic stub
 * gateway, launched as its own OS process for the cross-process end-to-end test.
 *
 *   PORT=0 STEPS=6 STEP_MS=100 npx tsx tests/e2e/fixtures/serve-stub-entry.ts
 *
 * The real CLI (`milkie serve --agent x.md`) resolves a real LLM gateway from
 * the agent's model config — not usable in a hermetic test. This fixture wires
 * the same createServeServer/runServeServer plumbing to a stub stepper gateway
 * so the HTTP+SSE surface, readiness signal, interrupt/resume, and SIGTERM
 * shutdown can be exercised across a true process boundary. The CLI arg/loader
 * layer (serveMain) is covered separately by unit tests.
 */
import { createServeServer, runServeServer } from '../../../src/cli/serve.js'
import { Milkie } from '../../../src/runtime/Milkie.js'
import { MemoryStore } from '../../../src/store/MemoryStore.js'
import { MemoryEventStore } from '../../../src/trace/MemoryEventStore.js'
import { BroadcastingEventStore } from '../../../src/trace/BroadcastingEventStore.js'
import type { AgentConfig } from '../../../src/types/agent.js'
import type { IModelGateway, ModelRequest, ModelResponse, ModelEvent } from '../../../src/types/model.js'
import type { ToolDefinition } from '../../../src/types/tool.js'

const totalSteps = Number(process.env['STEPS'] ?? 6)
const stepMs = Number(process.env['STEP_MS'] ?? 100)
let counter = 0

const workStep: ToolDefinition = {
  name: 'work_step', description: 'one unit of work',
  inputSchema: { type: 'object', properties: { stepId: { type: 'number' } }, required: ['stepId'] },
  handler: async (input: unknown) => {
    const { stepId } = input as { stepId: number }
    await new Promise<void>(r => setTimeout(r, stepMs))
    return { stepId, done: true }
  },
}

const nextRound = (): { done: boolean; stepId: number } => {
  if (counter >= totalSteps) return { done: true, stepId: 0 }
  counter++
  return { done: false, stepId: counter }
}

const gateway: IModelGateway = {
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const { done, stepId } = nextRound()
    if (done) return { content: [{ type: 'text', text: 'all done' }], toolCalls: [], finishReason: 'end_turn' }
    return { content: [{ type: 'tool_use', id: `w${stepId}`, name: 'work_step', input: { stepId } }], toolCalls: [{ id: `w${stepId}`, name: 'work_step', input: { stepId } }], finishReason: 'tool_use' }
  },
  async *stream(_req: ModelRequest): AsyncIterable<ModelEvent> {
    const { done, stepId } = nextRound()
    if (done) {
      // final turn: two token deltas so consumers can prove token-level streaming
      yield { type: 'message_delta', data: { text: 'all ' } }
      yield { type: 'message_delta', data: { text: 'done' } }
      return
    }
    yield { type: 'tool_call_start', data: { toolCallId: `w${stepId}`, name: 'work_step' } }
    yield { type: 'tool_call_done',  data: { toolCallId: `w${stepId}`, input: { stepId } } }
  },
}

const agent: AgentConfig = {
  agentId: 'stepper', version: '1.0.0', systemPrompt: 'step until done',
  fsm: { states: [{ name: 'react', type: 'llm', max_iterations: totalSteps + 5 }] },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
}

const broadcaster = new BroadcastingEventStore(new MemoryEventStore())
const milkie = new Milkie({ stateStore: new MemoryStore(), eventStore: broadcaster, gateway, tools: [workStep] })
milkie.registerAgent(agent)

const server = createServeServer({ milkie, agentId: 'stepper', broadcaster })
void runServeServer(server, { port: Number(process.env['PORT'] ?? 0) })
