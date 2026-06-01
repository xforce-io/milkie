/**
 * Live smoke test — requires VOLCENGINE_TOKEN and VOLCENGINE_API_BASE env vars.
 * Run with: npx jest volcengine.smoke --no-coverage
 */

import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { ConsoleRecorder } from '../trajectory/ConsoleRecorder'
import type { AgentConfig } from '../types/agent'

const SKIP = !process.env['VOLCENGINE_TOKEN'] || !process.env['VOLCENGINE_API_BASE']

const reactAgent: AgentConfig = {
  agentId:      'smoke-react',
  version:      '1.0.0',
  systemPrompt: 'You are a helpful assistant. Answer concisely.',
  fsm: {
    states: [{ name: 'react', type: 'llm', max_iterations: 5 }],
  },
  model: {
    provider: 'volcengine',
    model:    'doubao-seed-2.0-lite',
    adapter:  'openai-compatible',
  },
}

describe('VolcEngine smoke test', () => {
  let milkie: Milkie

  beforeAll(() => {
    milkie = new Milkie({ stateStore: new MemoryStore() })
    milkie.registerAgent(reactAgent)
  })

  const live = SKIP ? it.skip : it

  live('single-turn text response', async () => {
    const result = await milkie.invoke({
      agentId: 'smoke-react',
      goal:    'answer a simple question',
      input:   '1+1=?',
    })

    console.log('output:', result.output)
    expect(result.status).toBe('completed')
    expect(result.output).toMatch(/2/)
  }, 30000)

  live('tool calling — think then answer', async () => {
    const result = await milkie.invoke({
      agentId: 'smoke-react',
      goal:    'solve a problem',
      input:   'What is the square root of 144?',
    })

    console.log('output:', result.output)
    expect(result.status).toBe('completed')
    expect(result.output.length).toBeGreaterThan(0)
  }, 30000)
})
