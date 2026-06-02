import { makeCorpusTools } from '../tools/corpus-tools'
import fs from 'fs'
import os from 'os'
import path from 'path'

describe('corpus-tools (sandboxed)', () => {
  let tmpDir: string
  let listDir: ReturnType<typeof makeCorpusTools>['list_dir']
  let readFile: ReturnType<typeof makeCorpusTools>['read_file']
  let grep: ReturnType<typeof makeCorpusTools>['grep']

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'corpus-tools-'))
    fs.writeFileSync(path.join(tmpDir, 'a.txt'), 'hello world\nfoo bar\nbaz quux\n')
    fs.writeFileSync(path.join(tmpDir, 'b.txt'), 'second file\nhello again\n')
    fs.mkdirSync(path.join(tmpDir, 'sub'))
    fs.writeFileSync(path.join(tmpDir, 'sub', 'nested.txt'), 'deep content\nhello deep\n')

    const tools = makeCorpusTools(tmpDir)
    listDir  = tools.list_dir
    readFile = tools.read_file
    grep     = tools.grep
  })
  afterEach(() => { fs.rmSync(tmpDir, { recursive: true, force: true }) })

  // ─── list_dir ────────────────────────────────────────────────────────
  it('list_dir at root returns top-level files and subdirs', async () => {
    const result = await listDir({ relPath: '.' }) as { entries: Array<{ name: string; kind: string }> }
    const names = result.entries.map(e => e.name).sort()
    expect(names).toEqual(['a.txt', 'b.txt', 'sub'])
  })

  it('list_dir handles nested path', async () => {
    const result = await listDir({ relPath: 'sub' }) as { entries: Array<{ name: string; kind: string }> }
    expect(result.entries.map(e => e.name)).toEqual(['nested.txt'])
  })

  it('list_dir rejects path escaping corpus root', async () => {
    await expect(listDir({ relPath: '../escape' })).rejects.toThrow(/outside corpus/i)
  })

  it('list_dir rejects absolute path escaping corpus root', async () => {
    await expect(listDir({ relPath: '/etc' })).rejects.toThrow(/outside corpus/i)
  })

  // ─── read_file ───────────────────────────────────────────────────────
  it('read_file returns file content with line numbers', async () => {
    const result = await readFile({ relPath: 'a.txt' }) as { content: string; lines: number }
    expect(result.lines).toBe(3)
    expect(result.content).toContain('hello world')
    expect(result.content).toContain('foo bar')
  })

  it('read_file rejects path escaping corpus root', async () => {
    await expect(readFile({ relPath: '../../etc/passwd' })).rejects.toThrow(/outside corpus/i)
  })

  it('read_file rejects missing file', async () => {
    await expect(readFile({ relPath: 'nonexistent.txt' })).rejects.toThrow(/ENOENT|not found/i)
  })

  // ─── grep ────────────────────────────────────────────────────────────
  it('grep returns matches across files with file:line:text', async () => {
    const result = await grep({ pattern: 'hello' }) as { matches: Array<{ file: string; line: number; text: string }> }
    expect(result.matches.length).toBe(3)
    const files = result.matches.map(m => m.file).sort()
    expect(files).toEqual(['a.txt', 'b.txt', 'sub/nested.txt'])
  })

  it('grep is case-sensitive by default', async () => {
    const result = await grep({ pattern: 'HELLO' }) as { matches: Array<unknown> }
    expect(result.matches).toHaveLength(0)
  })

  it('grep with caseInsensitive: true returns case-insensitive matches', async () => {
    const result = await grep({ pattern: 'HELLO', caseInsensitive: true }) as { matches: Array<unknown> }
    expect(result.matches.length).toBeGreaterThan(0)
  })

  it('grep matches the pattern on EVERY consecutive line (no g-flag lastIndex skipping)', async () => {
    // Regression: a /g/ RegExp reused across lines via .test() carries lastIndex
    // state, so the same pattern on consecutive lines gets skipped on alternate
    // lines. All three "foo" lines must be matched, not just lines 1 and 3.
    fs.writeFileSync(path.join(tmpDir, 'repeat.txt'), 'foo\nfoo\nfoo\n')
    const result = await grep({ pattern: 'foo' }) as { matches: Array<{ file: string; line: number }> }
    const repeatHits = result.matches.filter(m => m.file === 'repeat.txt').map(m => m.line)
    expect(repeatHits).toEqual([1, 2, 3])
  })

  it('grep limits results to maxMatches (default 50)', async () => {
    const lines = Array.from({ length: 100 }, (_, i) => `match-line-${i}`).join('\n')
    fs.writeFileSync(path.join(tmpDir, 'big.txt'), lines + '\n')
    const result = await grep({ pattern: 'match-line' }) as { matches: Array<unknown>; truncated: boolean }
    expect(result.matches.length).toBe(50)
    expect(result.truncated).toBe(true)
  })
})
