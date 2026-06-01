// src/__tests__/standardAgentLayer.test.ts
import { Milkie } from '../runtime/Milkie'
import fs from 'fs'
import os from 'os'
import path from 'path'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'

class StubGateway implements IModelGateway {
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.responses.shift(); if (!r) throw new Error('stub exhausted'); return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}
const textResp = (s: string): ModelResponse => ({ content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn' })

const NO_MODEL_AGENT = `---
agentId: nomodel
version: 0.0.1
fsm:
  states:
    - name: react
      type: llm
      on: { DONE: end }
    - name: end
      type: action
      terminal: true
---
say hi`

function writeAgent(dir: string, name: string, body: string): string {
  const p = path.join(dir, name)
  fs.writeFileSync(p, body)
  return p
}

describe('#89 parseConfig: model optional', () => {
  let tmp: string
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-std-')) })
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }) })

  it('loads an agent template with NO model block (model is undefined)', () => {
    const file = writeAgent(tmp, 'tpl.md', `---
agentId: tpl
version: 0.0.1
fsm:
  states:
    - name: s
      type: llm
---
sys prompt`)
    const milkie = new Milkie()
    const cfg = milkie.loadAgentFile(file)
    expect(cfg.agentId).toBe('tpl')
    expect(cfg.model).toBeUndefined()
  })

  it('still throws when a model block is present but incomplete', () => {
    const file = writeAgent(tmp, 'bad.md', `---
agentId: bad
fsm:
  states:
    - name: s
      type: llm
model:
  provider: x
---
p`)
    const milkie = new Milkie()
    expect(() => milkie.loadAgentFile(file)).toThrow(/model/)
  })
})

describe('#89 resolveGateway: no-model agent', () => {
  let tmp: string
  beforeEach(() => { tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-rg-')) })
  afterEach(() => { fs.rmSync(tmp, { recursive: true, force: true }) })

  it('runs a no-model agent using the gateway override', async () => {
    const milkie = new Milkie({ gateway: new StubGateway([textResp('hi')]) })
    milkie.loadAgentFile(writeAgent(tmp, 'a.md', NO_MODEL_AGENT))
    const r = await milkie.invoke({ agentId: 'nomodel', goal: 'g', input: 'i' })
    expect(r.status).toBe('completed')
  })

  it('errors clearly when a no-model agent has neither gateway nor defaultModel', async () => {
    const milkie = new Milkie()
    milkie.loadAgentFile(writeAgent(tmp, 'a.md', NO_MODEL_AGENT))
    // resolveGateway is called synchronously near the top of invoke; it should surface
    // a clear error. If invoke rejects, use rejects.toThrow; if invoke swallows it into
    // a status:'error' result, assert on result.status/message instead. Pick whichever
    // the code actually does and make the test assert it.
    await expect(milkie.invoke({ agentId: 'nomodel', goal: 'g', input: 'i' }))
      .rejects.toThrow(/gateway.*defaultModel|no model/i)
  })
})
