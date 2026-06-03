// #126: AgentConfig gains an open `models` map of named tiers alongside the
// default `model`. parseConfig must read a frontmatter `models:` block, validate
// each tier like the default model, and stay backward-compatible (no `models`
// block → undefined; unknown fields ignored).
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import fs from 'fs'
import path from 'path'
import os from 'os'

let tmpDir: string
beforeEach(() => { tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-cfg-')) })
afterEach(() => { fs.rmSync(tmpDir, { recursive: true, force: true }) })

function writeAgent(body: string): string {
  const file = path.join(tmpDir, 'agent.md')
  fs.writeFileSync(file, body)
  return file
}

function newMilkie(): Milkie {
  return new Milkie({ stateStore: new MemoryStore() })
}

describe('#126 parseConfig — named model tiers', () => {
  it('parses a models block into config.models alongside the default model', () => {
    const file = writeAgent(`---
agentId: tiered
fsm:
  states: []
model:
  provider: volcengine
  model: default-model
  adapter: openai
models:
  fast:
    provider: volcengine
    model: qwen-turbo
    adapter: openai
    baseUrl: https://fast.example
---
sys`)
    const config = newMilkie().loadAgentFile(file)
    expect(config.model).toMatchObject({ provider: 'volcengine', model: 'default-model', adapter: 'openai' })
    expect(config.models).toEqual({
      fast: { provider: 'volcengine', model: 'qwen-turbo', adapter: 'openai', baseUrl: 'https://fast.example' },
    })
  })

  it('no models block → config.models is undefined (backward compatible)', () => {
    const file = writeAgent(`---
agentId: plain
fsm:
  states: []
model:
  provider: stub
  model: stub
  adapter: stub
---
sys`)
    const config = newMilkie().loadAgentFile(file)
    expect(config.models).toBeUndefined()
  })

  it('a tier missing provider/model/adapter is rejected, like the default model', () => {
    const file = writeAgent(`---
agentId: broken
fsm:
  states: []
models:
  fast:
    provider: volcengine
    adapter: openai
---
sys`)
    expect(() => newMilkie().loadAgentFile(file)).toThrow(/provider, model, adapter/)
  })
})
