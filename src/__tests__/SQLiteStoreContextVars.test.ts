import { SQLiteStore } from '../store/SQLiteStore'
import * as os from 'os'
import * as path from 'path'
import * as fs from 'fs'

// #83 acceptance §7: cross-process semantics — two store instances over the SAME
// sqlite file stand in for two processes. A var written by one is visible to the other.
describe('SQLiteStore context vars cross-instance (#83)', () => {
  let dbPath: string

  beforeEach(() => {
    dbPath = path.join(os.tmpdir(), `milkie-ctxvar-${process.pid}-${Date.now()}.sqlite`)
  })

  afterEach(() => {
    for (const f of [dbPath, `${dbPath}-wal`, `${dbPath}-shm`]) {
      if (fs.existsSync(f)) fs.unlinkSync(f)
    }
  })

  it('a var written by one instance is visible to another on the same file', async () => {
    const writer = new SQLiteStore({ path: dbPath })
    await writer.init()
    await writer.set('context:c1:var:workspace_instructions', '用中文')
    await writer.set('context:c1:var:session_id', 's-9')

    // a separate instance over the same file = another process
    const reader = new SQLiteStore({ path: dbPath })
    await reader.init()
    expect(await reader.get('context:c1:var:workspace_instructions')).toBe('用中文')

    const list = await reader.list('context:c1:var:')
    const obj = Object.fromEntries(list.map(e => [e.key, e.value]))
    expect(obj).toEqual({
      'context:c1:var:workspace_instructions': '用中文',
      'context:c1:var:session_id': 's-9',
    })

    writer.close()
    reader.close()
  })

  it('list respects prefix isolation across contexts and skips expired', async () => {
    const s = new SQLiteStore({ path: dbPath })
    await s.init()
    await s.set('context:c1:var:a', 1)
    await s.set('context:c2:var:b', 2)
    await s.set('context:c1:var:tmp', 99, -1)  // already expired

    const c1 = await s.list('context:c1:var:')
    expect(c1.map(e => e.key)).toEqual(['context:c1:var:a'])
    s.close()
  })
})
