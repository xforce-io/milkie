// Unit tests for the repair-ticketing agent's pure tool handlers (#185). No live
// model, no credentials — these pin commit_description's trust-boundary behavior:
// default to the verbatim turn (multi-turn), opt into a clean fault substring for
// one-shot via the optional `description` param.

import { WorkingMemory } from '../../../../src/store/WorkingMemory.js'
import type { ToolContext } from '../../../../src/types/tool.js'
import { makeCommitDescriptionTool } from '../agent.js'

const tool = makeCommitDescriptionTool()

function makeCtx(currentTurn: string): { ctx: ToolContext; wm: WorkingMemory } {
  const wm = new WorkingMemory()
  const ctx = { workingMemory: wm, currentTurn } as unknown as ToolContext
  return { ctx, wm }
}

describe('commit_description — optional description param (#185)', () => {
  it('defaults to ctx.currentTurn verbatim when no param is given (multi-turn 行为不变)', async () => {
    const { ctx, wm } = makeCtx('三楼会议室投影仪无法开机')
    const out = await tool.handler({}, ctx)
    expect(wm.get('description')).toBe('三楼会议室投影仪无法开机')
    expect(out).toEqual({ description: '三楼会议室投影仪无法开机' })
  })

  it('uses the supplied description param for a one-shot turn (整句不污染)', async () => {
    // One-shot: the whole turn carries levels + fault; the model passes only the
    // clean fault substring so `description` is not contaminated by level原话.
    const { ctx, wm } = makeCtx('总部主楼IT网络部王芳，三楼会议室投影仪无法开机')
    const out = await tool.handler({ description: '三楼会议室投影仪无法开机' }, ctx)
    expect(wm.get('description')).toBe('三楼会议室投影仪无法开机')
    expect(out).toEqual({ description: '三楼会议室投影仪无法开机' })
  })

  it('falls back to currentTurn when the param is blank/whitespace', async () => {
    const { ctx, wm } = makeCtx('服务器宕机了')
    await tool.handler({ description: '   ' }, ctx)
    expect(wm.get('description')).toBe('服务器宕机了')
  })
})
