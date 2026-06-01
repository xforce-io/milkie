// src/__tests__/standardAgentLayer.test.ts
import { Milkie } from '../runtime/Milkie'
import fs from 'fs'
import os from 'os'
import path from 'path'

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
