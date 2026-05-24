# agent-docs-qa Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `examples/agent-docs-qa/` — a Q&A agent over a vendored 三国演义 corpus, with a single-file Node `http` server + SSE + vanilla HTML/JS UI that exposes live trace observation and skill-loading verification. End-to-end exercise of milkie's `skill_request` → epoch boundary → `loadedSkills` evolution; live trace stream wires the example as both daily-use agent and milkie's first real visual demo of agent runtime capabilities.

**Architecture:** Bare Node `http` server hosts a single `Milkie` instance with the `sanguo-researcher` agent (base skill + on-demand `verifier`). Per-request `POST /chat` calls `Milkie.invoke({contextId, input})` for multi-turn continuation; `BroadcastingEventStore` decorates `JsonlEventStore` to publish each appended event to per-contextId SSE subscribers. Frontend is a single `index.html` with inline CSS+JS: left-panel chat, right-panel trace timeline (rendered from SSE stream for live conversations, REST for past), `loadedSkills` change highlighted. URL state `?context=<id>` lets the UI pivot between conversations without rewriting any data layer.

**Tech Stack:** TypeScript, Node `http`/`fs`/`path` (stdlib only), `EventSource` (browser native), milkie SDK (`Milkie.invoke`, `IEventStore`, `ToolDefinition`). No new npm dependencies. Vanilla HTML/CSS/JS — no build pipeline.

---

## File structure

**New:**

```
examples/agent-docs-qa/
├── README.md                                  # how to run + recommended questions
├── .gitignore                                 # ignore .milkie/runs + state.sqlite
├── corpus/
│   ├── chapter-01-桃园三结义.txt              # vendored from Wikisource
│   ├── chapter-37-三顾茅庐.txt
│   ├── chapter-49-赤壁借东风.txt
│   ├── chapter-50-华容道.txt
│   └── chapter-66-单刀赴会.txt
├── agents/
│   └── sanguo-researcher.md                   # AgentConfig md (base + verifier skill)
├── tools/
│   └── corpus-tools.ts                        # list_dir / read_file / grep (sandboxed)
├── trace/
│   ├── broadcast-event-store.ts               # IEventStore decorator + SSE pub/sub
│   └── conversation-scanner.ts                # scan .milkie/runs/ → group by contextId
├── server.ts                                  # http server + POST /chat + 3 GET endpoints
├── public/
│   └── index.html                             # vanilla HTML + inline <style>/<script>
└── package.json                               # placeholder (no deps; uses ../../src)
```

**Test files** (under `examples/agent-docs-qa/__tests__/`):

```
examples/agent-docs-qa/__tests__/
├── corpus-tools.test.ts                       # sandbox + handler correctness
├── broadcast-event-store.test.ts              # double-write + per-context routing
├── conversation-scanner.test.ts               # group runIds by contextId, sort by ts
└── server.test.ts                             # integration: POST /chat + GET endpoints + SSE
```

**Modify:** none — example is fully self-contained.

---

## Task 1: Scaffold example directory + .gitignore + package.json

**Files:**
- Create: `examples/agent-docs-qa/.gitignore`
- Create: `examples/agent-docs-qa/package.json`
- Create: `examples/agent-docs-qa/README.md` (skeleton — fleshed out in Task 11)

- [ ] **Step 1: Create `.gitignore`**

```gitignore
# Runtime state — re-created on demand
.milkie/state.sqlite
.milkie/state.sqlite-journal
.milkie/state.sqlite-wal
.milkie/state.sqlite-shm
.milkie/runs/

# Browser cached state (none today; future-proofing)
*.log
```

(Corpus IS committed — that's the example's vendored content. Runtime `.milkie/runs/` is NOT committed — every user generates their own conversations.)

- [ ] **Step 2: Create `package.json`**

```json
{
  "name": "agent-docs-qa-example",
  "version": "0.0.0",
  "private": true,
  "description": "milkie example — 三国 Q&A agent with skill loading + live trace UI",
  "scripts": {
    "start": "npx tsx server.ts"
  }
}
```

(No deps — reuses parent `package.json`'s deps via TypeScript path resolution.)

- [ ] **Step 3: Create README skeleton**

```markdown
# agent-docs-qa — 三国 Q&A with live trace observation

Runnable example demonstrating milkie's skill-loading + live trace
capabilities. See [design spec](../../docs/superpowers/specs/2026-05-24-agent-docs-qa-example-design.md).

(Full walkthrough lands in Task 11.)
```

- [ ] **Step 4: Commit**

```bash
git add examples/agent-docs-qa/.gitignore examples/agent-docs-qa/package.json examples/agent-docs-qa/README.md
git commit -m "scaffold(examples): agent-docs-qa directory + .gitignore + package.json"
```

---

## Task 2: Vendor 三国演义 corpus (5 chapters from Wikisource)

**Files:**
- Create: `examples/agent-docs-qa/corpus/chapter-01-桃园三结义.txt`
- Create: `examples/agent-docs-qa/corpus/chapter-37-三顾茅庐.txt`
- Create: `examples/agent-docs-qa/corpus/chapter-49-赤壁借东风.txt`
- Create: `examples/agent-docs-qa/corpus/chapter-50-华容道.txt`
- Create: `examples/agent-docs-qa/corpus/chapter-66-单刀赴会.txt`

- [ ] **Step 1: Fetch from Wikisource**

The 5 chapters are at:
- `https://zh.wikisource.org/wiki/三國演義/第001回` (第一回 宴桃園豪傑三結義 斬黃巾英雄首立功)
- `https://zh.wikisource.org/wiki/三國演義/第037回` (第三十七回 司馬徽再薦名士 劉玄德三顧草廬)
- `https://zh.wikisource.org/wiki/三國演義/第049回` (第四十九回 七星壇諸葛祭風 三江口周瑜縱火)
- `https://zh.wikisource.org/wiki/三國演義/第050回` (第五十回 諸葛亮智算華容 關雲長義釋曹操)
- `https://zh.wikisource.org/wiki/三國演義/第066回` (第六十六回 關雲長單刀赴會 伏皇后為國捐生)

For each chapter:
1. Fetch the page (e.g. `curl -L 'https://zh.wikisource.org/wiki/...?action=raw'` — Wikisource exposes raw wiki source via `?action=raw`)
2. Strip wiki markup: remove `{{...}}`, `[[...|...]]` → keep the text after `|`, `==...==` headings can stay as plain lines, `<ref>...</ref>` tags removed
3. Strip header noise (the navigation row at top), keep only the chapter text
4. Save as UTF-8, LF line endings (no BOM)
5. Filename: `chapter-NN-<标题关键词>.txt` (繁体 ok — leave as-is from source; OpenCC conversion is optional and adds dependency)

Target size: 10-20 KB per file; 50-80 KB total.

- [ ] **Step 2: Verify size + readability**

```bash
wc -c examples/agent-docs-qa/corpus/*.txt
file examples/agent-docs-qa/corpus/*.txt
head -3 examples/agent-docs-qa/corpus/chapter-01-桃园三结义.txt
```

Expected: each file 10-30 KB; file type `UTF-8 Unicode text`; head shows recognizable 三国演义 opening lines (e.g., `話說天下大勢，分久必合，合久必分...`).

- [ ] **Step 3: Commit**

```bash
git add examples/agent-docs-qa/corpus/
git commit -m "examples(agent-docs-qa): vendor 5 chapters of 三国演义 from Wikisource

Public domain (作者罗贯中, c.1330-1400). Source:
https://zh.wikisource.org/wiki/三國演義

Wiki markup stripped, UTF-8 LF normalized. Selected chapters cover
桃园三结义 / 三顾茅庐 / 赤壁 / 华容道 / 单刀赴会 — heavy character +
event cross-references for grep+read multi-turn reasoning demo."
```

---

## Task 3: Sandboxed corpus tools (list_dir / read_file / grep)

**Files:**
- Create: `examples/agent-docs-qa/tools/corpus-tools.ts`
- Test: `examples/agent-docs-qa/__tests__/corpus-tools.test.ts`

- [ ] **Step 1: Write failing test**

```typescript
// examples/agent-docs-qa/__tests__/corpus-tools.test.ts
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

  it('grep limits results to maxMatches (default 50)', async () => {
    // Add a file with 100 matches
    const lines = Array.from({ length: 100 }, (_, i) => `match-line-${i}`).join('\n')
    fs.writeFileSync(path.join(tmpDir, 'big.txt'), lines + '\n')
    const result = await grep({ pattern: 'match-line' }) as { matches: Array<unknown>; truncated: boolean }
    expect(result.matches.length).toBe(50)
    expect(result.truncated).toBe(true)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest examples/agent-docs-qa/__tests__/corpus-tools.test.ts`
Expected: FAIL — `Cannot find module '../tools/corpus-tools'`.

- [ ] **Step 3: Implement `corpus-tools.ts`**

```typescript
// examples/agent-docs-qa/tools/corpus-tools.ts
import { promises as fs } from 'fs'
import path from 'path'

/**
 * Build a sandboxed set of corpus tools rooted at `corpusRoot`.
 * Every tool resolves user-provided relPath against corpusRoot, then
 * verifies the resolved absolute path is still inside corpusRoot
 * (rejecting `..` escapes and absolute paths pointing elsewhere).
 */
export function makeCorpusTools(corpusRoot: string) {
  const root = path.resolve(corpusRoot)

  function resolveInsideRoot(relPath: string): string {
    const abs = path.resolve(root, relPath)
    if (abs !== root && !abs.startsWith(root + path.sep)) {
      throw new Error(`path "${relPath}" resolves outside corpus root`)
    }
    return abs
  }

  async function list_dir(input: unknown): Promise<unknown> {
    const { relPath } = input as { relPath: string }
    const abs = resolveInsideRoot(relPath)
    const entries = await fs.readdir(abs, { withFileTypes: true })
    return {
      entries: entries.map(e => ({
        name: e.name,
        kind: e.isDirectory() ? 'directory' : 'file',
      })),
    }
  }

  async function read_file(input: unknown): Promise<unknown> {
    const { relPath } = input as { relPath: string }
    const abs = resolveInsideRoot(relPath)
    const content = await fs.readFile(abs, 'utf-8')
    const lines = content.split('\n').length
    return { content, lines }
  }

  async function grep(input: unknown): Promise<unknown> {
    const { pattern, caseInsensitive = false } = input as { pattern: string; caseInsensitive?: boolean }
    const maxMatches = 50
    const flags = caseInsensitive ? 'gi' : 'g'
    const re = new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), flags)
    const matches: Array<{ file: string; line: number; text: string }> = []

    async function walk(dir: string): Promise<void> {
      if (matches.length >= maxMatches) return
      const entries = await fs.readdir(dir, { withFileTypes: true })
      for (const e of entries) {
        if (matches.length >= maxMatches) return
        const abs = path.join(dir, e.name)
        if (e.isDirectory()) {
          await walk(abs)
        } else if (e.isFile()) {
          const content = await fs.readFile(abs, 'utf-8')
          const lines = content.split('\n')
          for (let i = 0; i < lines.length; i++) {
            if (re.test(lines[i]!)) {
              matches.push({
                file: path.relative(root, abs),
                line: i + 1,
                text: lines[i]!.slice(0, 200),
              })
              if (matches.length >= maxMatches) break
            }
          }
        }
      }
    }
    await walk(root)
    return {
      matches,
      truncated: matches.length >= maxMatches,
    }
  }

  return { list_dir, read_file, grep }
}

/**
 * Build ToolDefinition objects ready for Milkie.registerTool. Wraps
 * makeCorpusTools handlers with the JSONSchema input contracts the agent
 * sees and uses.
 */
export function makeCorpusToolDefinitions(corpusRoot: string) {
  const t = makeCorpusTools(corpusRoot)
  return [
    {
      name:        'list_dir',
      description: 'List entries in a directory within the corpus.',
      inputSchema: {
        type: 'object',
        properties: { relPath: { type: 'string', description: 'Path relative to corpus root. Use "." for root.' } },
        required: ['relPath'],
      },
      handler: t.list_dir,
    },
    {
      name:        'read_file',
      description: 'Read full content of a file within the corpus.',
      inputSchema: {
        type: 'object',
        properties: { relPath: { type: 'string', description: 'Path relative to corpus root.' } },
        required: ['relPath'],
      },
      handler: t.read_file,
    },
    {
      name:        'grep',
      description: 'Search for a pattern across all files in the corpus. Returns up to 50 matches.',
      inputSchema: {
        type: 'object',
        properties: {
          pattern:         { type: 'string', description: 'Literal string to search for (regex-escaped automatically).' },
          caseInsensitive: { type: 'boolean', description: 'Default false. Set true for case-insensitive search.' },
        },
        required: ['pattern'],
      },
      handler: t.grep,
    },
  ]
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest examples/agent-docs-qa/__tests__/corpus-tools.test.ts`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/agent-docs-qa/tools/corpus-tools.ts examples/agent-docs-qa/__tests__/corpus-tools.test.ts
git commit -m "feat(examples): sandboxed corpus tools (list_dir / read_file / grep)"
```

---

## Task 4: Agent definition (sanguo-researcher.md)

**Files:**
- Create: `examples/agent-docs-qa/agents/sanguo-researcher.md`

This task has no test of its own — the agent config is consumed by Task 5+ tests. Content task only.

- [ ] **Step 1: Create agent md**

```markdown
---
agentId: sanguo-researcher
version: 0.0.1
fsm:
  states:
    - name: respond
      type: llm
      instructions: |
        你是《三国演义》的研究助手。回答用户提出的关于《三国演义》的问题。

        语料库（corpus）放在 corpus/ 目录下，文件名形如
        chapter-NN-标题.txt。

        工作流程：
        - 用 list_dir({ relPath: "." }) 查看有哪些章节
        - 用 grep({ pattern: "..." }) 找出相关章节（关键词可以是人名、地名、事件、对白）
        - 用 read_file({ relPath: "chapter-XX-..." }) 读相关段落
        - 回答时尽量给出 chapter:行号 形式的引用

        重要：当用户对你的回答表达怀疑（"你确定吗" / "再确认下" /
        "verify" / "are you sure" / "真的吗" 等），调用
        skill_request({ name: "verifier" }) 进入下一 epoch 的严格
        验证模式，并在本轮回答里告知用户"已申请加载 verifier，下一轮
        将严格 verify"。

        verifier 是一次性的——同一会话内不要反复 request；如果用户
        再次怀疑、且 verifier 已加载，直接以严格模式重新验证即可。
      tools: [list_dir, read_file, grep, skill_request]
model:
  provider: openai
  model: gpt-4o-mini
  adapter: openai-compatible
skills:
  verifier: "0.1.0"
skillInstructions:
  verifier: |
    你已进入 verifier 模式。

    重新读你前一轮回答里引用过的所有原文段落，把每一条陈述分类：
    - (a) 直接 supported by text：原文措辞与你陈述高度一致，给出 chapter:行号 citation
    - (b) inferred from text：基于原文推理得出（明示推理链）
    - (c) unfounded：原文没有支撑，承认错误并更正

    严格判断——只要措辞与原文不严格匹配，就退到 (b) 或 (c)。
    宁可保守，不要为了好看给 (a)。
---
你是《三国演义》的研究助手。Corpus 锁定在本目录的 corpus/ 子目录。
```

- [ ] **Step 2: Verify YAML parses + agent loads**

Quick sanity check via Node:
```bash
npx tsx -e "
import { Milkie } from './src/runtime/Milkie'
const m = new Milkie()
const cfg = m.loadAgentFile('examples/agent-docs-qa/agents/sanguo-researcher.md')
console.log(JSON.stringify({
  agentId: cfg.agentId,
  toolsCount: cfg.fsm.states[0].tools.length,
  skills: Object.keys(cfg.skills ?? {}),
  skillInstructionsKeys: Object.keys(cfg.skillInstructions ?? {}),
}, null, 2))
"
```
Expected:
```
{
  "agentId": "sanguo-researcher",
  "toolsCount": 4,
  "skills": ["verifier"],
  "skillInstructionsKeys": ["verifier"]
}
```

- [ ] **Step 3: Commit**

```bash
git add examples/agent-docs-qa/agents/sanguo-researcher.md
git commit -m "feat(examples): sanguo-researcher AgentConfig with verifier skill"
```

---

## Task 5: BroadcastingEventStore (double-write + per-context routing)

**Files:**
- Create: `examples/agent-docs-qa/trace/broadcast-event-store.ts`
- Test: `examples/agent-docs-qa/__tests__/broadcast-event-store.test.ts`

- [ ] **Step 1: Write failing test**

```typescript
// examples/agent-docs-qa/__tests__/broadcast-event-store.test.ts
import { BroadcastingEventStore } from '../trace/broadcast-event-store'
import { MemoryEventStore } from '../../../src/trace/MemoryEventStore'
import type { Event } from '../../../src/trace/types'

const startedEvent = (runId: string, contextId: string): Event => ({
  id: `${runId}-start`, runId, type: 'agent.run.started', actor: 'runtime', timestamp: 1,
  payload: { agentId: 'a', goal: 'g', input: 'i', contextId },
})
const llmEvent = (runId: string, id: string): Event => ({
  id, runId, type: 'llm.requested', actor: 'runtime', timestamp: 2,
  payload: { request: {}, requestHash: 'h' },
})

describe('BroadcastingEventStore', () => {
  it('forwards append to inner store', async () => {
    const inner = new MemoryEventStore()
    const store = new BroadcastingEventStore(inner)
    await store.append(startedEvent('r1', 'ctx1'))
    expect(await inner.readByRunId('r1')).toHaveLength(1)
  })

  it('broadcasts subsequent events to subscribers of the matching contextId', async () => {
    const store = new BroadcastingEventStore(new MemoryEventStore())
    const received: Event[] = []
    const unsubscribe = store.subscribe('ctx1', e => { received.push(e) })

    await store.append(startedEvent('r1', 'ctx1'))
    await store.append(llmEvent('r1', 'evt-2'))

    expect(received).toHaveLength(2)
    expect(received[0]!.id).toBe('r1-start')
    expect(received[1]!.id).toBe('evt-2')

    unsubscribe()
  })

  it('does NOT broadcast events of a different contextId to subscriber', async () => {
    const store = new BroadcastingEventStore(new MemoryEventStore())
    const receivedA: Event[] = []
    const receivedB: Event[] = []
    store.subscribe('ctxA', e => { receivedA.push(e) })
    store.subscribe('ctxB', e => { receivedB.push(e) })

    await store.append(startedEvent('runA', 'ctxA'))
    await store.append(llmEvent('runA', 'a-evt-2'))
    await store.append(startedEvent('runB', 'ctxB'))
    await store.append(llmEvent('runB', 'b-evt-2'))

    expect(receivedA.map(e => e.id).sort()).toEqual(['a-evt-2', 'runA-start'])
    expect(receivedB.map(e => e.id).sort()).toEqual(['b-evt-2', 'runB-start'])
  })

  it('unsubscribe stops further deliveries', async () => {
    const store = new BroadcastingEventStore(new MemoryEventStore())
    const received: Event[] = []
    const unsub = store.subscribe('ctx1', e => { received.push(e) })

    await store.append(startedEvent('r1', 'ctx1'))
    unsub()
    await store.append(llmEvent('r1', 'evt-2'))

    expect(received).toHaveLength(1)
  })

  it('multiple subscribers on same contextId all receive', async () => {
    const store = new BroadcastingEventStore(new MemoryEventStore())
    const a: Event[] = []
    const b: Event[] = []
    store.subscribe('ctx1', e => { a.push(e) })
    store.subscribe('ctx1', e => { b.push(e) })

    await store.append(startedEvent('r1', 'ctx1'))

    expect(a).toHaveLength(1)
    expect(b).toHaveLength(1)
  })

  it('forwards readByRunId / readRange to inner', async () => {
    const inner = new MemoryEventStore()
    const store = new BroadcastingEventStore(inner)
    await store.append(startedEvent('r1', 'ctx1'))
    await store.append(llmEvent('r1', 'evt-2'))

    expect(await store.readByRunId('r1')).toHaveLength(2)
    expect(await store.readRange('r1', 1)).toHaveLength(1)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest examples/agent-docs-qa/__tests__/broadcast-event-store.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `broadcast-event-store.ts`**

```typescript
// examples/agent-docs-qa/trace/broadcast-event-store.ts
import type { IEventStore } from '../../../src/trace/EventStore.js'
import type { Event } from '../../../src/trace/types.js'

type Subscriber = (event: Event) => void

/**
 * IEventStore decorator that:
 *  1. Delegates persistence to an inner store (typically JsonlEventStore).
 *  2. Tracks runId → contextId mappings via agent.run.started events
 *     observed during append().
 *  3. Broadcasts each appended event to subscribers registered for the
 *     event's contextId.
 *
 * The runId → contextId cache survives only as long as this instance
 * (i.e., the server process). Historic conversation queries that need to
 * resolve contextIds from disk go through conversation-scanner.ts.
 */
export class BroadcastingEventStore implements IEventStore {
  private readonly subscribers: Map<string, Set<Subscriber>> = new Map()
  private readonly contextIdByRunId: Map<string, string> = new Map()

  constructor(private readonly inner: IEventStore) {}

  async append(event: Event): Promise<void> {
    await this.inner.append(event)

    if (event.type === 'agent.run.started') {
      const payload = event.payload as { contextId: string }
      this.contextIdByRunId.set(event.runId, payload.contextId)
    }

    const contextId = this.contextIdByRunId.get(event.runId)
    if (contextId) {
      const subs = this.subscribers.get(contextId)
      if (subs) for (const cb of subs) cb(event)
    }
  }

  async readByRunId(runId: string): Promise<Event[]> {
    return this.inner.readByRunId(runId)
  }

  async readRange(runId: string, fromIndex: number, count?: number): Promise<Event[]> {
    return this.inner.readRange(runId, fromIndex, count)
  }

  /**
   * Subscribe to live appended events for a given contextId.
   * Returns an unsubscribe function. Caller must invoke it on SSE close
   * to prevent memory leaks.
   */
  subscribe(contextId: string, cb: Subscriber): () => void {
    let set = this.subscribers.get(contextId)
    if (!set) {
      set = new Set()
      this.subscribers.set(contextId, set)
    }
    set.add(cb)
    return () => { set!.delete(cb) }
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest examples/agent-docs-qa/__tests__/broadcast-event-store.test.ts`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/agent-docs-qa/trace/broadcast-event-store.ts examples/agent-docs-qa/__tests__/broadcast-event-store.test.ts
git commit -m "feat(examples): BroadcastingEventStore — IEventStore decorator + per-context pub/sub"
```

---

## Task 6: Conversation scanner (group runIds by contextId from disk)

**Files:**
- Create: `examples/agent-docs-qa/trace/conversation-scanner.ts`
- Test: `examples/agent-docs-qa/__tests__/conversation-scanner.test.ts`

- [ ] **Step 1: Write failing test**

```typescript
// examples/agent-docs-qa/__tests__/conversation-scanner.test.ts
import { scanConversations, readEventsForContext } from '../trace/conversation-scanner'
import fs from 'fs'
import os from 'os'
import path from 'path'

function writeRun(
  baseDir: string,
  runId: string,
  contextId: string,
  startedAt: number,
  completedStatus?: string,
): void {
  const events: Array<Record<string, unknown>> = [
    {
      id: `${runId}-s`, runId, type: 'agent.run.started', actor: 'runtime', timestamp: startedAt,
      payload: { agentId: 'sanguo-researcher', goal: 'g', input: 'i', contextId },
    },
    {
      id: `${runId}-l1`, runId, type: 'llm.requested', actor: 'runtime', timestamp: startedAt + 1,
      payload: { request: {}, requestHash: 'h' },
    },
  ]
  if (completedStatus) {
    events.push({
      id: `${runId}-c`, runId, type: 'agent.run.completed', actor: 'runtime', timestamp: startedAt + 2,
      payload: { status: completedStatus },
    })
  }
  fs.writeFileSync(
    path.join(baseDir, `${runId}.jsonl`),
    events.map(e => JSON.stringify(e)).join('\n') + '\n',
  )
}

describe('conversation-scanner', () => {
  let tmpDir: string
  beforeEach(() => { tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'conv-scanner-')) })
  afterEach(() => { fs.rmSync(tmpDir, { recursive: true, force: true }) })

  it('scanConversations returns empty for empty dir', async () => {
    expect(await scanConversations(tmpDir)).toEqual([])
  })

  it('scanConversations returns empty for nonexistent dir', async () => {
    expect(await scanConversations(path.join(tmpDir, 'nope'))).toEqual([])
  })

  it('scanConversations groups multiple runs by contextId', async () => {
    writeRun(tmpDir, 'run1', 'ctxA', 100, 'completed')
    writeRun(tmpDir, 'run2', 'ctxA', 200, 'completed')
    writeRun(tmpDir, 'run3', 'ctxB', 150, 'completed')

    const convs = await scanConversations(tmpDir)
    expect(convs).toHaveLength(2)

    const ctxA = convs.find(c => c.contextId === 'ctxA')!
    expect(ctxA.runIds.sort()).toEqual(['run1', 'run2'])
    expect(ctxA.agentId).toBe('sanguo-researcher')

    const ctxB = convs.find(c => c.contextId === 'ctxB')!
    expect(ctxB.runIds).toEqual(['run3'])
  })

  it('scanConversations sorts by most-recent startedAt descending', async () => {
    writeRun(tmpDir, 'old',  'ctxOld',  100, 'completed')
    writeRun(tmpDir, 'newer', 'ctxNew', 500, 'completed')
    writeRun(tmpDir, 'mid',  'ctxMid',  300, 'completed')

    const convs = await scanConversations(tmpDir)
    expect(convs.map(c => c.contextId)).toEqual(['ctxNew', 'ctxMid', 'ctxOld'])
  })

  it('scanConversations marks "active" when latest run lacks completed event', async () => {
    writeRun(tmpDir, 'r1', 'ctxLive', 100)  // no completed
    const convs = await scanConversations(tmpDir)
    expect(convs[0]!.status).toBe('active')
  })

  it('scanConversations reports "completed" when latest run has completed event', async () => {
    writeRun(tmpDir, 'r1', 'ctxDone', 100, 'completed')
    const convs = await scanConversations(tmpDir)
    expect(convs[0]!.status).toBe('completed')
  })

  it('readEventsForContext returns events of all matching runIds in timestamp order', async () => {
    writeRun(tmpDir, 'run1', 'ctxA', 100, 'completed')
    writeRun(tmpDir, 'run2', 'ctxA', 200, 'completed')
    writeRun(tmpDir, 'run3', 'ctxB', 150, 'completed')

    const events = await readEventsForContext(tmpDir, 'ctxA')
    expect(events.filter(e => e.runId === 'run3')).toHaveLength(0)
    expect(events.length).toBe(6)  // 3 events × 2 runs

    const timestamps = events.map(e => e.timestamp)
    const sorted = [...timestamps].sort((a, b) => a - b)
    expect(timestamps).toEqual(sorted)
  })

  it('readEventsForContext returns empty for unknown contextId', async () => {
    writeRun(tmpDir, 'r1', 'ctxA', 100, 'completed')
    expect(await readEventsForContext(tmpDir, 'ctxNonexistent')).toEqual([])
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest examples/agent-docs-qa/__tests__/conversation-scanner.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `conversation-scanner.ts`**

```typescript
// examples/agent-docs-qa/trace/conversation-scanner.ts
import { promises as fs } from 'fs'
import path from 'path'
import type { Event } from '../../../src/trace/types.js'

export interface ConversationSummary {
  contextId: string
  agentId:   string
  startedAt: number       // earliest startedAt across runs
  status:    'active' | 'completed' | 'error' | 'interrupted'
  runIds:    string[]     // in startedAt order
  eventCount: number
}

interface RunMeta {
  runId:     string
  contextId: string
  agentId:   string
  startedAt: number
  status:    'active' | 'completed' | 'error' | 'interrupted'
  eventCount: number
}

async function listRunFiles(runsDir: string): Promise<string[]> {
  try {
    const entries = await fs.readdir(runsDir)
    return entries.filter(n => n.endsWith('.jsonl'))
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return []
    throw err
  }
}

async function readMeta(runsDir: string, file: string): Promise<RunMeta | null> {
  try {
    const content = await fs.readFile(path.join(runsDir, file), 'utf-8')
    const lines = content.split('\n').filter(l => l.length > 0)
    if (lines.length === 0) return null

    const first = JSON.parse(lines[0]!) as Event
    if (first.type !== 'agent.run.started') return null

    const startedPayload = first.payload as { agentId: string; contextId: string }
    let status: RunMeta['status'] = 'active'

    for (let i = lines.length - 1; i >= 0; i--) {
      const evt = JSON.parse(lines[i]!) as Event
      if (evt.type === 'agent.run.completed') {
        const p = evt.payload as { status: string }
        status = (p.status as RunMeta['status']) ?? 'completed'
        break
      }
    }

    return {
      runId:      first.runId,
      contextId:  startedPayload.contextId,
      agentId:    startedPayload.agentId,
      startedAt:  first.timestamp,
      status,
      eventCount: lines.length,
    }
  } catch {
    return null
  }
}

/**
 * Scan the runs directory and group runs into conversations by contextId.
 * Sorted by most-recent startedAt descending. Active conversations (no
 * completed event in the latest run) get status='active'.
 */
export async function scanConversations(runsDir: string): Promise<ConversationSummary[]> {
  const files = await listRunFiles(runsDir)
  const metas: RunMeta[] = []
  for (const f of files) {
    const m = await readMeta(runsDir, f)
    if (m) metas.push(m)
  }

  const grouped = new Map<string, RunMeta[]>()
  for (const m of metas) {
    const arr = grouped.get(m.contextId) ?? []
    arr.push(m)
    grouped.set(m.contextId, arr)
  }

  const conversations: ConversationSummary[] = []
  for (const [contextId, runs] of grouped) {
    runs.sort((a, b) => a.startedAt - b.startedAt)
    const latest = runs[runs.length - 1]!
    conversations.push({
      contextId,
      agentId:    latest.agentId,
      startedAt:  runs[0]!.startedAt,
      status:     latest.status,
      runIds:     runs.map(r => r.runId),
      eventCount: runs.reduce((sum, r) => sum + r.eventCount, 0),
    })
  }

  conversations.sort((a, b) => b.startedAt - a.startedAt)
  return conversations
}

/**
 * Read all events for a contextId across its constituent runs, sorted
 * by timestamp ascending (in-conversation chronological order).
 */
export async function readEventsForContext(runsDir: string, contextId: string): Promise<Event[]> {
  const conversations = await scanConversations(runsDir)
  const target = conversations.find(c => c.contextId === contextId)
  if (!target) return []

  const all: Event[] = []
  for (const runId of target.runIds) {
    const content = await fs.readFile(path.join(runsDir, `${runId}.jsonl`), 'utf-8')
    const events = content
      .split('\n')
      .filter(l => l.length > 0)
      .map(l => JSON.parse(l) as Event)
    all.push(...events)
  }
  all.sort((a, b) => a.timestamp - b.timestamp)
  return all
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest examples/agent-docs-qa/__tests__/conversation-scanner.test.ts`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/agent-docs-qa/trace/conversation-scanner.ts examples/agent-docs-qa/__tests__/conversation-scanner.test.ts
git commit -m "feat(examples): conversation-scanner — group runIds by contextId from disk"
```

---

## Task 7: HTTP server — POST /chat + GET /conversations + GET /conversation/:id/events

**Files:**
- Create: `examples/agent-docs-qa/server.ts`
- Test: `examples/agent-docs-qa/__tests__/server.test.ts`

This task introduces the server skeleton with three non-SSE endpoints. SSE comes in Task 8.

- [ ] **Step 1: Write failing test**

```typescript
// examples/agent-docs-qa/__tests__/server.test.ts
import { startServer, stopServer } from '../server'
import type { Server } from 'http'
import type { IModelGateway, ModelRequest, ModelResponse } from '../../../src/types/model'
import fs from 'fs'
import os from 'os'
import path from 'path'

class StubGateway implements IModelGateway {
  constructor(private readonly responses: ModelResponse[]) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    const r = this.responses.shift()
    if (!r) throw new Error('StubGateway exhausted')
    return r
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> { yield* [] }
}

const text = (s: string): ModelResponse => ({
  content: [{ type: 'text', text: s }], toolCalls: [], finishReason: 'end_turn',
})

async function get(url: string): Promise<{ status: number; body: string }> {
  const res = await fetch(url)
  return { status: res.status, body: await res.text() }
}

async function postJson(url: string, body: unknown): Promise<{ status: number; body: string }> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  return { status: res.status, body: await res.text() }
}

describe('server — REST endpoints', () => {
  let server: Server
  let baseUrl: string
  let exampleDir: string

  beforeEach(async () => {
    exampleDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-docs-qa-server-'))
    fs.mkdirSync(path.join(exampleDir, '.milkie', 'runs'), { recursive: true })

    server = await startServer({
      port:        0,                     // auto-assign
      exampleDir,
      gateway:     new StubGateway([text('hello from stub')]),
      agentFile:   path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
      corpusRoot:  path.join(__dirname, '..', 'corpus'),
    })
    const addr = server.address()
    if (!addr || typeof addr === 'string') throw new Error('server address unavailable')
    baseUrl = `http://localhost:${addr.port}`
  })

  afterEach(async () => {
    await stopServer(server)
    fs.rmSync(exampleDir, { recursive: true, force: true })
  })

  it('GET / returns the index.html', async () => {
    const r = await get(`${baseUrl}/`)
    expect(r.status).toBe(200)
    expect(r.body).toContain('<!doctype html>')
    expect(r.body).toContain('agent playground')
  })

  it('POST /chat with no contextId mints a new one and returns runId + contextId', async () => {
    const r = await postJson(`${baseUrl}/chat`, { input: 'hi' })
    expect(r.status).toBe(200)
    const body = JSON.parse(r.body) as { runId: string; contextId: string; status: string }
    expect(body.runId).toMatch(/^[0-9a-f-]{36}$/)
    expect(body.contextId).toMatch(/^[0-9a-f-]{36}$/)
    expect(body.status).toBe('completed')
  })

  it('POST /chat with same contextId twice continues the same conversation', async () => {
    await stopServer(server)
    server = await startServer({
      port: 0, exampleDir,
      gateway: new StubGateway([text('first'), text('second')]),
      agentFile:  path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
      corpusRoot: path.join(__dirname, '..', 'corpus'),
    })
    const addr = server.address()
    baseUrl = `http://localhost:${(addr as { port: number }).port}`

    const a = JSON.parse((await postJson(`${baseUrl}/chat`, { input: 'q1' })).body) as { contextId: string; runId: string }
    const b = JSON.parse((await postJson(`${baseUrl}/chat`, { input: 'q2', contextId: a.contextId })).body) as { contextId: string; runId: string }
    expect(b.contextId).toBe(a.contextId)
    expect(b.runId).not.toBe(a.runId)
  })

  it('GET /conversations returns empty list initially', async () => {
    const r = await get(`${baseUrl}/conversations`)
    expect(r.status).toBe(200)
    expect(JSON.parse(r.body)).toEqual({ conversations: [] })
  })

  it('GET /conversations lists prior chats after POST /chat', async () => {
    await postJson(`${baseUrl}/chat`, { input: 'hi' })
    const r = await get(`${baseUrl}/conversations`)
    const body = JSON.parse(r.body) as { conversations: Array<{ contextId: string }> }
    expect(body.conversations).toHaveLength(1)
    expect(body.conversations[0]!.contextId).toMatch(/^[0-9a-f-]{36}$/)
  })

  it('GET /conversation/:id/events returns all events for the context in time order', async () => {
    const chat = JSON.parse((await postJson(`${baseUrl}/chat`, { input: 'hi' })).body) as { contextId: string }
    const r = await get(`${baseUrl}/conversation/${chat.contextId}/events`)
    expect(r.status).toBe(200)
    const body = JSON.parse(r.body) as { events: Array<{ type: string; timestamp: number }> }
    expect(body.events.length).toBeGreaterThan(0)
    const timestamps = body.events.map(e => e.timestamp)
    expect([...timestamps].sort((a, b) => a - b)).toEqual(timestamps)
  })

  it('GET /conversation/:id/events returns 404 for unknown contextId', async () => {
    const r = await get(`${baseUrl}/conversation/nonexistent/events`)
    expect(r.status).toBe(404)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest examples/agent-docs-qa/__tests__/server.test.ts`
Expected: FAIL — `Cannot find module '../server'`.

- [ ] **Step 3: Implement `server.ts` (skeleton + REST endpoints; SSE in Task 8)**

```typescript
// examples/agent-docs-qa/server.ts
import http, { type IncomingMessage, type ServerResponse, type Server } from 'http'
import { promises as fs } from 'fs'
import { existsSync, mkdirSync } from 'fs'
import path from 'path'
import { v4 as uuidv4 } from 'uuid'
import { Milkie } from '../../src/runtime/Milkie.js'
import { MemoryStore } from '../../src/store/MemoryStore.js'
import { JsonlEventStore } from '../../src/trace/JsonlEventStore.js'
import { createGateway } from '../../src/gateway/GatewayFactory.js'
import type { IModelGateway } from '../../src/types/model.js'
import { BroadcastingEventStore } from './trace/broadcast-event-store.js'
import { scanConversations, readEventsForContext } from './trace/conversation-scanner.js'
import { makeCorpusToolDefinitions } from './tools/corpus-tools.js'

export interface ServerConfig {
  port:        number
  exampleDir:  string
  gateway?:    IModelGateway      // override (test injection)
  agentFile:   string
  corpusRoot:  string
}

interface ServerState {
  milkie:      Milkie
  eventStore:  BroadcastingEventStore
  runsDir:     string
  publicDir:   string
}

let state: ServerState | undefined

async function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    req.on('data', c => chunks.push(c as Buffer))
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')))
    req.on('error', reject)
  })
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const json = JSON.stringify(body)
  res.writeHead(status, { 'content-type': 'application/json' })
  res.end(json)
}

async function handleChat(req: IncomingMessage, res: ServerResponse, s: ServerState): Promise<void> {
  const raw = await readBody(req)
  const { input, contextId } = JSON.parse(raw) as { input: string; contextId?: string }
  if (!input) { sendJson(res, 400, { error: 'input required' }); return }

  const ctxId = contextId ?? uuidv4()
  const result = await s.milkie.invoke({
    agentId:   'sanguo-researcher',
    goal:      input,
    input,
    contextId: ctxId,
  })

  sendJson(res, 200, {
    runId:     result.agentRunId,
    contextId: ctxId,
    status:    result.status,
    output:    result.output,
  })
}

async function handleListConversations(res: ServerResponse, s: ServerState): Promise<void> {
  const conversations = await scanConversations(s.runsDir)
  sendJson(res, 200, { conversations })
}

async function handleGetConversationEvents(
  res: ServerResponse, s: ServerState, contextId: string,
): Promise<void> {
  const events = await readEventsForContext(s.runsDir, contextId)
  if (events.length === 0) { sendJson(res, 404, { error: 'conversation not found' }); return }
  sendJson(res, 200, { events })
}

async function serveStatic(res: ServerResponse, filePath: string): Promise<void> {
  try {
    const content = await fs.readFile(filePath, 'utf-8')
    const ext = path.extname(filePath)
    const ctype = ext === '.html' ? 'text/html; charset=utf-8' : 'text/plain'
    res.writeHead(200, { 'content-type': ctype })
    res.end(content)
  } catch {
    res.writeHead(404).end()
  }
}

export async function startServer(config: ServerConfig): Promise<Server> {
  const runsDir = path.join(config.exampleDir, '.milkie', 'runs')
  if (!existsSync(runsDir)) mkdirSync(runsDir, { recursive: true })

  const eventStore = new BroadcastingEventStore(new JsonlEventStore(runsDir))
  const milkie     = new Milkie({
    stateStore: new MemoryStore(),
    gateway:    config.gateway,   // when omitted, falls back to per-agent createGateway
    eventStore,
  })

  for (const tool of makeCorpusToolDefinitions(config.corpusRoot)) {
    milkie.registerTool(tool)
  }
  milkie.loadAgentFile(config.agentFile)

  const publicDir = path.join(config.exampleDir, 'public')
  // publicDir may not exist in test setup that uses tmpDir; serve from
  // the example's actual public dir when present, else fall back to
  // the package's public/ relative to this file.
  const effectivePublicDir = existsSync(publicDir)
    ? publicDir
    : path.resolve(path.dirname(new URL(import.meta.url).pathname), 'public')

  state = { milkie, eventStore, runsDir, publicDir: effectivePublicDir }

  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', 'http://localhost')
      const route = url.pathname

      if (req.method === 'POST' && route === '/chat')
        return handleChat(req, res, state!)
      if (req.method === 'GET' && route === '/conversations')
        return handleListConversations(res, state!)

      const convMatch = route.match(/^\/conversation\/([^/]+)\/events$/)
      if (req.method === 'GET' && convMatch) {
        return handleGetConversationEvents(res, state!, decodeURIComponent(convMatch[1]!))
      }

      if (req.method === 'GET' && (route === '/' || route === '/index.html')) {
        return serveStatic(res, path.join(state!.publicDir, 'index.html'))
      }

      res.writeHead(404).end()
    } catch (err) {
      sendJson(res, 500, { error: (err as Error).message })
    }
  })

  await new Promise<void>(resolve => server.listen(config.port, () => resolve()))
  return server
}

export async function stopServer(server: Server): Promise<void> {
  state = undefined
  await new Promise<void>(resolve => server.close(() => resolve()))
}

// CLI entry: only runs when invoked directly via `npx tsx server.ts`
const isMain = import.meta.url === `file://${process.argv[1]}`
if (isMain) {
  const PORT = Number(process.env.PORT ?? 7878)
  const EXAMPLE_DIR = path.dirname(new URL(import.meta.url).pathname)
  startServer({
    port:       PORT,
    exampleDir: EXAMPLE_DIR,
    agentFile:  path.join(EXAMPLE_DIR, 'agents', 'sanguo-researcher.md'),
    corpusRoot: path.join(EXAMPLE_DIR, 'corpus'),
  }).then(() => {
    console.log(`agent-docs-qa playground at http://localhost:${PORT}`)
  })
}
```

Note: `state` is module-level because tests bind it via startServer and need it accessible inside route handlers; tests serialize via beforeEach/afterEach so no cross-test pollution.

- [ ] **Step 4: Create a minimal placeholder `public/index.html`** so the GET / test passes

```html
<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>agent playground</title></head>
<body><h1>agent playground</h1><p>(placeholder — full UI in Task 10)</p></body>
</html>
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx jest examples/agent-docs-qa/__tests__/server.test.ts`
Expected: PASS (7 tests).

Note: tests that POST /chat trigger a real `Milkie.invoke` against the agent file. The agent's FSM will call grep/list_dir tools which actually scan the corpus dir (passed via `corpusRoot`). The stub gateway returns a single 'hello from stub' so the FSM should accept the response and complete.

If the agent attempts tool calls (the stub response doesn't include toolCalls so it shouldn't), confirm by reading the test output. If issues: the gateway can return additional canned responses to satisfy the FSM.

- [ ] **Step 6: Commit**

```bash
git add examples/agent-docs-qa/server.ts examples/agent-docs-qa/public/index.html examples/agent-docs-qa/__tests__/server.test.ts
git commit -m "feat(examples): http server skeleton — POST /chat + GET /conversations + GET /conversation/:id/events"
```

---

## Task 8: GET /conversation/:id/stream (SSE — past events then live)

**Files:**
- Modify: `examples/agent-docs-qa/server.ts`
- Test: `examples/agent-docs-qa/__tests__/server.test.ts` (extend)

- [ ] **Step 1: Write failing test**

Append to `examples/agent-docs-qa/__tests__/server.test.ts`:

```typescript
describe('server — SSE stream', () => {
  let server: Server
  let baseUrl: string
  let exampleDir: string

  beforeEach(async () => {
    exampleDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agent-docs-qa-sse-'))
    fs.mkdirSync(path.join(exampleDir, '.milkie', 'runs'), { recursive: true })
    server = await startServer({
      port: 0, exampleDir,
      gateway: new StubGateway([text('first'), text('second')]),
      agentFile:  path.join(__dirname, '..', 'agents', 'sanguo-researcher.md'),
      corpusRoot: path.join(__dirname, '..', 'corpus'),
    })
    const addr = server.address()
    baseUrl = `http://localhost:${(addr as { port: number }).port}`
  })
  afterEach(async () => {
    await stopServer(server)
    fs.rmSync(exampleDir, { recursive: true, force: true })
  })

  it('SSE delivers past events on connect then closes for completed conversation', async () => {
    // Record a conversation first
    const first = JSON.parse((await postJson(`${baseUrl}/chat`, { input: 'q1' })).body) as { contextId: string }

    // Connect to SSE for this contextId
    const res = await fetch(`${baseUrl}/conversation/${first.contextId}/stream`)
    expect(res.headers.get('content-type')).toContain('text/event-stream')

    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let received = ''

    // Read until stream closes or 1-second timeout
    const timeout = setTimeout(() => reader.cancel(), 1000)
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      received += decoder.decode(value)
    }
    clearTimeout(timeout)

    // Each SSE message: "data: <json>\n\n"
    const messages = received.split('\n\n').filter(s => s.startsWith('data:'))
    expect(messages.length).toBeGreaterThan(0)
    const events = messages.map(m => JSON.parse(m.replace(/^data: /, '')))
    expect(events.some((e: { type: string }) => e.type === 'agent.run.started')).toBe(true)
    expect(events.some((e: { type: string }) => e.type === 'agent.run.completed')).toBe(true)
  }, 5_000)

  it('SSE delivers live events appended after subscribe', async () => {
    // Mint a contextId via a first chat
    const first = JSON.parse((await postJson(`${baseUrl}/chat`, { input: 'q1' })).body) as { contextId: string }

    // Open SSE
    const res = await fetch(`${baseUrl}/conversation/${first.contextId}/stream`)
    const reader = res.body!.getReader()
    const decoder = new TextDecoder()

    // Drain initial catch-up
    const drainOnce = async (): Promise<string> => {
      const { value } = await reader.read()
      return decoder.decode(value ?? new Uint8Array())
    }
    let buffer = ''
    while (!buffer.includes('agent.run.completed')) buffer += await drainOnce()

    // Trigger a second invoke on the same context — should push new events
    postJson(`${baseUrl}/chat`, { input: 'q2', contextId: first.contextId })

    // Continue reading; expect a new agent.run.started for the second run
    const timeout = setTimeout(() => reader.cancel(), 2000)
    while (!buffer.match(/agent\.run\.started.+?agent\.run\.started/s)) {
      const chunk = await drainOnce()
      if (!chunk) break
      buffer += chunk
    }
    clearTimeout(timeout)

    // Should see TWO agent.run.started in total (one per invoke)
    const startedCount = (buffer.match(/"type":"agent\.run\.started"/g) ?? []).length
    expect(startedCount).toBe(2)
  }, 10_000)
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest examples/agent-docs-qa/__tests__/server.test.ts -t "SSE"`
Expected: FAIL — 404 (route not implemented).

- [ ] **Step 3: Add SSE handler to `server.ts`**

Add this function before `startServer`:

```typescript
async function handleSseStream(
  req: IncomingMessage, res: ServerResponse, s: ServerState, contextId: string,
): Promise<void> {
  res.writeHead(200, {
    'content-type':  'text/event-stream',
    'cache-control': 'no-cache',
    'connection':    'keep-alive',
    // SSE pre-flight: large initial buffer helps with browsers
    'x-accel-buffering': 'no',
  })

  // 1. Catch-up: send all past events for this context
  const past = await readEventsForContext(s.runsDir, contextId)
  for (const e of past) {
    res.write(`data: ${JSON.stringify(e)}\n\n`)
  }

  // 2. Subscribe to live events
  const unsubscribe = s.eventStore.subscribe(contextId, (event) => {
    if (!res.writableEnded) res.write(`data: ${JSON.stringify(event)}\n\n`)
  })

  // 3. If no past events AND no current active run for this contextId,
  //    close immediately so the client doesn't hang on an unknown contextId.
  if (past.length === 0) {
    unsubscribe()
    res.end()
    return
  }

  // 4. Cleanup on client disconnect
  req.on('close', () => {
    unsubscribe()
    if (!res.writableEnded) res.end()
  })
}
```

Add the route match in the http.createServer handler, before the 404:

```typescript
const sseMatch = route.match(/^\/conversation\/([^/]+)\/stream$/)
if (req.method === 'GET' && sseMatch) {
  return handleSseStream(req, res, state!, decodeURIComponent(sseMatch[1]!))
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest examples/agent-docs-qa/__tests__/server.test.ts -t "SSE"`
Expected: PASS (2 tests). Full file: 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add examples/agent-docs-qa/server.ts examples/agent-docs-qa/__tests__/server.test.ts
git commit -m "feat(examples): SSE /conversation/:id/stream — past events + live subscription"
```

---

## Task 9: Frontend HTML/CSS skeleton + conversation picker

**Files:**
- Modify: `examples/agent-docs-qa/public/index.html` (replace placeholder with real UI)

This task is content-heavy with little testable logic. Smoke-test by loading the page in a browser; manual verification in Task 11.

- [ ] **Step 1: Replace `examples/agent-docs-qa/public/index.html`**

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>milkie agent playground — 三国 Q&A</title>
  <style>
    :root {
      --bg: #f7f7f8;
      --panel: #ffffff;
      --border: #e5e5e7;
      --text: #1c1c1e;
      --muted: #6e6e73;
      --accent: #5b3ec9;
      --skill-highlight: #fff4e0;
    }
    * { box-sizing: border-box }
    html, body { margin: 0; padding: 0; height: 100% }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC", sans-serif;
      background: var(--bg); color: var(--text); font-size: 14px;
    }
    header {
      padding: 12px 20px; background: var(--panel); border-bottom: 1px solid var(--border);
      display: flex; align-items: center; gap: 16px;
    }
    header h1 { margin: 0; font-size: 16px; font-weight: 600 }
    header .controls { margin-left: auto; display: flex; gap: 8px; align-items: center }
    header select, header button {
      font: inherit; padding: 4px 10px; border: 1px solid var(--border);
      border-radius: 4px; background: var(--panel); cursor: pointer;
    }
    header button { background: var(--accent); color: white; border-color: var(--accent) }

    main { display: flex; height: calc(100vh - 49px) }
    #chat { flex: 1; display: flex; flex-direction: column; border-right: 1px solid var(--border); background: var(--panel) }
    #chat-log { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px }
    .msg { padding: 8px 12px; border-radius: 8px; max-width: 80%; line-height: 1.5; white-space: pre-wrap }
    .msg.user      { background: #e8f0fe; align-self: flex-end }
    .msg.assistant { background: #f0f0f2; align-self: flex-start }
    .msg .speaker { font-size: 11px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase }
    #chat-input { display: flex; padding: 12px; border-top: 1px solid var(--border); gap: 8px }
    #chat-input input { flex: 1; padding: 8px 12px; font: inherit; border: 1px solid var(--border); border-radius: 4px }
    #chat-input input:disabled { background: #f5f5f7; color: var(--muted) }
    #chat-input button { padding: 8px 16px; background: var(--accent); color: white; border: none; border-radius: 4px; cursor: pointer; font: inherit }
    #chat-input button:disabled { opacity: 0.5; cursor: not-allowed }

    #trace { flex: 1; display: flex; flex-direction: column; background: var(--bg) }
    #trace-header { padding: 8px 16px; background: var(--panel); border-bottom: 1px solid var(--border); font-size: 12px; color: var(--muted) }
    #trace-timeline { flex: 1; overflow-y: auto; padding: 12px }
    .entry {
      padding: 6px 12px; margin-bottom: 4px; background: var(--panel); border-radius: 4px;
      border: 1px solid transparent; cursor: pointer; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px;
      transition: background 0.2s ease, border-color 0.2s ease;
    }
    .entry:hover { border-color: var(--border) }
    .entry.skill-loaded { background: var(--skill-highlight); border-color: #f0c060; font-weight: 600 }
    .entry .icon { display: inline-block; width: 16px; color: var(--accent) }
    .entry.tool .icon { color: #2563eb }
    .entry.lifecycle .icon { color: var(--muted) }
    .entry .ts { color: var(--muted); margin-left: 8px; font-size: 11px }

    #payload-detail {
      max-height: 35vh; overflow-y: auto; padding: 12px;
      background: #1c1c1e; color: #f5f5f7;
      font-family: ui-monospace, monospace; font-size: 11px; white-space: pre-wrap;
      border-top: 1px solid var(--border);
    }
    #payload-detail:empty { display: none }
  </style>
</head>
<body>
  <header>
    <h1>milkie agent playground — 三国 Q&A</h1>
    <div class="controls">
      <select id="conversation-picker">
        <option value="">(new conversation)</option>
      </select>
      <button id="new-chat">+ new chat</button>
    </div>
  </header>
  <main>
    <section id="chat">
      <div id="chat-log"></div>
      <form id="chat-input">
        <input type="text" placeholder="问《三国演义》相关问题..." autocomplete="off" />
        <button type="submit">发送</button>
      </form>
    </section>
    <section id="trace">
      <div id="trace-header">Trace timeline</div>
      <div id="trace-timeline"></div>
      <div id="payload-detail"></div>
    </section>
  </main>

  <script>
  /* Task 10 will fill this. */
  </script>
</body>
</html>
```

- [ ] **Step 2: Smoke-test by loading the page**

```bash
cd examples/agent-docs-qa
PORT=7878 npx tsx server.ts &
sleep 1
curl -s http://localhost:7878/ | head -5
# Expected: <!doctype html> ... <title>milkie agent playground — 三国 Q&A</title>
kill %1
```

(or just open in browser; the page renders empty chat + empty trace + dropdown with "(new conversation)".)

- [ ] **Step 3: Commit**

```bash
git add examples/agent-docs-qa/public/index.html
git commit -m "feat(examples): frontend HTML/CSS skeleton — chat + trace + payload layout"
```

---

## Task 10: Frontend JS — wire chat input, conversation picker, trace render, skill-load highlight

**Files:**
- Modify: `examples/agent-docs-qa/public/index.html` (fill the `<script>` block)

- [ ] **Step 1: Fill the `<script>` block**

Replace `/* Task 10 will fill this. */` with:

```javascript
(function () {
  // ─── State ─────────────────────────────────────────────────────────
  let currentContextId = new URLSearchParams(location.search).get('context') || null
  let eventSource = null
  let loadedSkillsByRun = {}  // runId → array of loadedSkills (latest snapshot)

  // ─── DOM ───────────────────────────────────────────────────────────
  const $log     = document.getElementById('chat-log')
  const $input   = document.querySelector('#chat-input input')
  const $form    = document.getElementById('chat-input')
  const $submit  = document.querySelector('#chat-input button')
  const $picker  = document.getElementById('conversation-picker')
  const $newBtn  = document.getElementById('new-chat')
  const $tl      = document.getElementById('trace-timeline')
  const $detail  = document.getElementById('payload-detail')
  const $traceHd = document.getElementById('trace-header')

  // ─── Utilities ─────────────────────────────────────────────────────
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;')
  }
  function setUrlContext(ctxId) {
    const url = new URL(location.href)
    if (ctxId) url.searchParams.set('context', ctxId)
    else url.searchParams.delete('context')
    history.replaceState(null, '', url.toString())
  }

  // ─── Conversation picker ───────────────────────────────────────────
  async function refreshPicker() {
    const res = await fetch('/conversations')
    const data = await res.json()
    const cur = $picker.value
    $picker.innerHTML = '<option value="">(new conversation)</option>'
    for (const c of data.conversations) {
      const opt = document.createElement('option')
      opt.value = c.contextId
      opt.textContent = `${c.contextId.slice(0, 8)}… · ${c.status} · ${c.runIds.length} turns`
      $picker.appendChild(opt)
    }
    $picker.value = currentContextId ?? ''
  }
  $picker.addEventListener('change', () => {
    const v = $picker.value
    currentContextId = v || null
    setUrlContext(currentContextId)
    if (currentContextId) loadConversation(currentContextId)
    else resetUI()
  })
  $newBtn.addEventListener('click', () => {
    currentContextId = null
    setUrlContext(null)
    $picker.value = ''
    resetUI()
  })

  // ─── UI reset ──────────────────────────────────────────────────────
  function resetUI() {
    $log.innerHTML = ''
    $tl.innerHTML = ''
    $detail.innerHTML = ''
    $input.disabled = false
    $submit.disabled = false
    $traceHd.textContent = 'Trace timeline (new conversation — send a message to start)'
    if (eventSource) { eventSource.close(); eventSource = null }
    loadedSkillsByRun = {}
  }

  // ─── Render an event into trace timeline ───────────────────────────
  function renderEvent(event) {
    const div = document.createElement('div')
    div.className = `entry ${kindClass(event.type)}`
    const icon = iconFor(event.type)
    const summary = summaryFor(event)
    div.innerHTML = `<span class="icon">${icon}</span>${esc(summary)}<span class="ts">${event.timestamp}</span>`

    // Skill-loading highlight: agent.run.started events whose loadedSkills
    // differs from the previous run's snapshot for this contextId.
    if (event.type === 'agent.run.started') {
      const skills = (event.payload && event.payload.loadedSkills) || []
      const prevSkills = Object.values(loadedSkillsByRun).pop() || []
      if (JSON.stringify(skills) !== JSON.stringify(prevSkills)) {
        div.classList.add('skill-loaded')
        div.title = `loadedSkills changed: [${prevSkills.join(', ')}] → [${skills.join(', ')}]`
      }
      loadedSkillsByRun[event.runId] = skills
    }

    // Click → show payload
    div.addEventListener('click', () => {
      $detail.textContent = JSON.stringify(event, null, 2)
    })

    $tl.appendChild(div)
    autoScroll()
  }

  function kindClass(type) {
    if (type.startsWith('llm.'))  return 'llm'
    if (type.startsWith('tool.')) return 'tool'
    return 'lifecycle'
  }
  function iconFor(type) {
    if (type.startsWith('llm.'))  return '◆'
    if (type.startsWith('tool.')) return '▣'
    return '●'
  }
  function summaryFor(event) {
    const t = event.type
    if (t === 'agent.run.started')   return `run started · ${(event.payload && event.payload.input || '').slice(0, 60)}`
    if (t === 'agent.run.completed') return `run completed · ${event.payload && event.payload.status}`
    if (t === 'llm.requested')       return 'LLM request'
    if (t === 'llm.responded')       return 'LLM response'
    if (t === 'tool.requested')      return `tool: ${event.payload && event.payload.toolName}`
    if (t === 'tool.responded')      return `tool: ${event.payload && event.payload.toolName} → ok`
    if (t === 'clock.read')          return `clock.read · ${event.payload && event.payload.value}`
    if (t === 'uuid.generated')      return `uuid.generated · ${(event.payload && event.payload.value || '').slice(0, 8)}…`
    return t
  }

  function renderChatMessage(role, text) {
    const div = document.createElement('div')
    div.className = `msg ${role}`
    div.innerHTML = `<div class="speaker">${role}</div>${esc(text)}`
    $log.appendChild(div)
    $log.scrollTop = $log.scrollHeight
  }

  function autoScroll() {
    // Only auto-scroll trace if user is near the bottom already
    const nearBottom = $tl.scrollHeight - $tl.scrollTop - $tl.clientHeight < 100
    if (nearBottom) $tl.scrollTop = $tl.scrollHeight
  }

  // ─── Load existing conversation (past events + maybe SSE) ──────────
  async function loadConversation(ctxId) {
    resetUI()
    $traceHd.textContent = `Trace timeline — conversation ${ctxId.slice(0, 8)}…`

    // 1. Pull conversation summary to know if it's active
    const convResp = await fetch('/conversations')
    const conv = (await convResp.json()).conversations.find(c => c.contextId === ctxId)
    const isActive = conv && conv.status === 'active'

    // 2. Pull past events to rebuild chat log + trace
    const eventsResp = await fetch(`/conversation/${ctxId}/events`)
    if (eventsResp.status === 404) return
    const { events } = await eventsResp.json()
    for (const e of events) {
      renderEvent(e)
      if (e.type === 'agent.run.started' && e.payload && e.payload.input) {
        renderChatMessage('user', e.payload.input)
      } else if (e.type === 'agent.run.completed' && e.payload && e.payload.lastTextOutput) {
        renderChatMessage('assistant', e.payload.lastTextOutput)
      }
    }

    // 3. Open SSE — handler ignores past events (we already have them)
    //    by tracking the last-seen event id; future events are appended.
    const lastEventId = events.length > 0 ? events[events.length - 1].id : null
    openSse(ctxId, lastEventId)

    // 4. If conversation ended, disable input
    if (!isActive) {
      $input.disabled = true
      $submit.disabled = true
      $input.placeholder = '此对话已结束。点 "+ new chat" 开始新对话。'
    }
  }

  function openSse(ctxId, lastEventId) {
    if (eventSource) eventSource.close()
    eventSource = new EventSource(`/conversation/${ctxId}/stream`)
    const seenIds = new Set()
    eventSource.onmessage = (msg) => {
      const event = JSON.parse(msg.data)
      if (seenIds.has(event.id)) return
      seenIds.add(event.id)
      // Skip events we already rendered from REST fetch
      if (lastEventId && event.timestamp <= 0) return  // safety guard
      renderEvent(event)
      if (event.type === 'agent.run.completed' && event.payload && event.payload.lastTextOutput) {
        renderChatMessage('assistant', event.payload.lastTextOutput)
      }
    }
    eventSource.onerror = () => {
      // SSE may close when server is done; that's expected
    }
  }

  // ─── Chat submission ───────────────────────────────────────────────
  $form.addEventListener('submit', async (ev) => {
    ev.preventDefault()
    const text = $input.value.trim()
    if (!text) return

    renderChatMessage('user', text)
    $input.value = ''
    $input.disabled = true
    $submit.disabled = true

    try {
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ input: text, contextId: currentContextId }),
      })
      const data = await resp.json()
      if (!currentContextId) {
        currentContextId = data.contextId
        setUrlContext(currentContextId)
        // Open SSE for the new conversation
        openSse(currentContextId, null)
      }
      // The completed-event handler in openSse will append the assistant response
    } catch (err) {
      renderChatMessage('assistant', `[error] ${err.message}`)
    } finally {
      $input.disabled = false
      $submit.disabled = false
      $input.focus()
      refreshPicker()  // newly-created conversation should appear
    }
  })

  // ─── Boot ──────────────────────────────────────────────────────────
  (async function () {
    await refreshPicker()
    if (currentContextId) {
      $picker.value = currentContextId
      await loadConversation(currentContextId)
    } else {
      resetUI()
    }
  })()
})()
```

- [ ] **Step 2: Smoke-test the page**

Start the server, open in browser, expected behavior:
- Dropdown shows "(new conversation)"
- Type a message in input + click 发送
- chat log shows your user message
- trace timeline starts filling with events (real-time as SSE arrives)
- After ~few seconds, assistant response appears in chat log
- Reload page (?context=… preserved) → conversation restored from past events

- [ ] **Step 3: Commit**

```bash
git add examples/agent-docs-qa/public/index.html
git commit -m "feat(examples): frontend JS — chat input + conversation picker + SSE trace render + skill-load highlight"
```

---

## Task 11: README walkthrough + manual E2E verification

**Files:**
- Modify: `examples/agent-docs-qa/README.md`

- [ ] **Step 1: Manual E2E walkthrough**

Set `OPENAI_API_KEY` (or compatible env var per `createGateway`). From the repo root:

```bash
npm run build
cd examples/agent-docs-qa
npx tsx server.ts
# Open http://localhost:7878 in browser
```

Test the following flow and note any issues for fixing in a follow-up commit:

1. **Cold start**: page loads, dropdown shows "(new conversation)", input is enabled.
2. **First question**: type "赤壁之战双方主帅是谁？" → submit. trace timeline starts filling. agent should grep for "赤壁", read at least one chapter, then respond. Response should mention 周瑜 (吴) / 曹操 (魏).
3. **Conversation continuity**: type "他们最后谁赢了？" → submit. Agent should answer based on prior context (the conversation is ongoing).
4. **Skill loading trigger**: type "你确定吗？" → submit. agent should call `skill_request("verifier")` (visible in trace as `tool.requested · skill_request`); next `agent.run.started` should be highlighted with the skill-loaded badge; subsequent answer should be in verifier mode (citation-rich, may correct itself).
5. **Conversation switching**: click "+ new chat", verify URL `?context=` drops, ask a different question. Then click the conversation picker dropdown — both conversations should be listed; pick the first one and verify chat log + trace restore from past events; input should be disabled with "此对话已结束".
6. **Past events click-to-detail**: click any trace entry → payload JSON appears in dark panel at bottom.

- [ ] **Step 2: Write the README**

Replace `examples/agent-docs-qa/README.md` with:

```markdown
# agent-docs-qa — 三国 Q&A with live trace observation

Runnable example: a Q&A agent over vendored 三国演义 corpus, with a small
local web server that exposes live trace observation and skill-loading
demonstration.

Design spec: [docs/superpowers/specs/2026-05-24-agent-docs-qa-example-design.md](../../docs/superpowers/specs/2026-05-24-agent-docs-qa-example-design.md)

## What this example demonstrates

1. **A real agent you'd actually use** — ask questions about 三国演义,
   get cited answers. Not a toy.
2. **Live trace observation** — UI shows the agent's full reasoning
   trajectory (grep, read, LLM call/response, lifecycle) as it happens.
3. **Skill loading (progressive disclosure)** — when you ask "you sure?",
   the agent calls `skill_request("verifier")` mid-conversation; the next
   turn loads a stricter verification skill into the context. The UI
   highlights the moment of skill load.
4. **Conversation browsing** — past conversations persist as JSONL on
   disk; the UI picker lets you revisit them (read-only).

## Setup

You need an OpenAI-compatible LLM endpoint. Set the API key in your env:

```bash
export OPENAI_API_KEY=sk-...
# or your provider's equivalent
```

The default agent uses `gpt-4o-mini` via the `openai-compatible` adapter.
To swap providers, edit `agents/sanguo-researcher.md` `model:` section.

## Run

```bash
# From repo root:
npm run build

cd examples/agent-docs-qa
npx tsx server.ts
# → agent-docs-qa playground at http://localhost:7878
```

Open `http://localhost:7878` in a browser.

## Try these

- "赤壁之战双方主帅是谁？"
- "诸葛亮第一次出场是在哪一回？"
- "关羽华容道为什么放走曹操？"
- After any answer: "你确定吗？" — triggers the verifier skill load

## What's in the corpus

Five chapters of 三国演义 from Wikisource (public domain):
- 第一回 桃园三结义
- 第三十七回 三顾茅庐
- 第四十九回 赤壁借东风
- 第五十回 华容道
- 第六十六回 单刀赴会

To use your own corpus: replace files in `corpus/` (filenames don't
matter; agent uses `list_dir` to discover them).

## Architecture (in one diagram)

```
browser (vanilla JS)        Node http server         Milkie SDK
─────────────────────       ──────────────────       ──────────
chat input ─POST /chat───►  milkie.invoke({         RecordingIOPort
                              contextId, input })   ↓
trace area ◄─SSE stream───  BroadcastingEvent       events flow into
   (auto-renders             Store: write to        JsonlEventStore
    incoming events)         JSONL + broadcast
                             to subscribers
conv picker ◄─/convs ────   scan .milkie/runs/
                             group by contextId
```

## File layout

```
.
├── agents/sanguo-researcher.md    # AgentConfig: base + verifier skill
├── corpus/                        # vendored chapters
├── tools/corpus-tools.ts          # sandboxed list_dir / read_file / grep
├── trace/
│   ├── broadcast-event-store.ts   # IEventStore + per-context pub/sub
│   └── conversation-scanner.ts    # group runIds by contextId from disk
├── server.ts                      # http + SSE
├── public/index.html              # vanilla UI (chat + trace + payload)
└── __tests__/                     # unit + integration tests
```

## What this does NOT do (intentional scope)

- **Resume past conversations** — past conversations are read-only;
  clicking one lets you browse but not continue. Use "+ new chat" to
  start fresh.
- **Multi-trajectory comparison view** — the data model and UI are
  ready for `?compare=A,B` (Phase 5 fork/diff), but the comparison
  view itself isn't built.
- **Multi-user / auth** — single-process localhost only.
- **Persistent active state across server restart** — server uses
  `MemoryStore`; live conversations are lost on restart. Trace JSONLs
  remain on disk and stay browseable.

## Related

- Story: [s-002](../../docs/stories/s-002-inspect-a-completed-run.md)
  (the static report this builds on)
- Story: [s-010](../../docs/stories/s-010-skill-versioned-load-and-ab-experiment.md)
  (the skill-loading capability this exercises)
- Architecture: [`Reference UI projection`](../../ARCHITECTURE.md)
```

- [ ] **Step 3: Commit**

```bash
git add examples/agent-docs-qa/README.md
git commit -m "docs(examples): agent-docs-qa README — walkthrough + scope + architecture"
```

---

## Self-review

**1. Spec coverage:**

| Spec section | Implementation task(s) |
|---|---|
| §1 目标与边界 — dogfood Q&A agent | Tasks 2, 4 (corpus + agent md) |
| §1 边界 — agent runtime 活体验收 | Tasks 4, 7, 10 (skill_request in agent md; trace UI in frontend) |
| §1 不做项（fork/resume/skill_list/RAG/生产形态/二个 skill） | All tasks honor — no fork views, no resume UI, no skill registry, no embedding |
| §2 决策表 (12 项) | All locked: A.1 corpus locked Task 2; A.2 grep+read tools Task 3; A.4 verifier skill Task 4; A.5 base+verifier Task 4; A.6 chat+trace layout Task 9; A.7 bare http+SSE Task 7+8; A.8 SSE Task 8; A.9 MemoryStore Task 7; A.10 contextId grouping Tasks 6, 10; A.11 multi-traj hooks Tasks 7-10 |
| §3 控制流图 | Tasks 5-8 (server) + 10 (frontend) |
| §4 example 目录 layout | Tasks 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (each file in layout has its task) |
| §4 agentConfig md | Task 4 |
| §4 server.ts | Tasks 7 (REST) + 8 (SSE) |
| §4 BroadcastingEventStore | Task 5 |
| §4 工具沙箱限定 corpus 根 | Task 3 (tested via "outside corpus" assertions) |
| §4 index.html | Tasks 9 (skeleton) + 10 (JS) |
| §5 trace timeline 渲染 | Task 10 (renderEvent + skill-loaded highlight) |
| §5 loadedSkills 高亮 | Task 10 (`div.classList.add('skill-loaded')`) |
| §5 payload detail click | Task 10 (`div.addEventListener('click', …)`) |
| §6 不变量 1 (multi-turn same-context) | Verified empirically during brainstorming; tested in Task 7 "POST /chat with same contextId" |
| §6 不变量 2 (skill_request → loadedSkills changed) | Manual E2E Task 11 step 4 |
| §6 不变量 3 (verifier 真进 system prompt) | Manual E2E Task 11 step 4 (trace payload inspection) |
| §6 不变量 4 (BroadcastingEventStore 双写) | Task 5 test |
| §6 不变量 5 (SSE 按 contextId 隔离) | Task 5 test "different contextId not delivered" + Task 8 SSE wiring |
| §6 不变量 6 (工具沙箱) | Task 3 tests "rejects path outside corpus" |
| §7 测试策略 unit / integration / E2E manual | Tasks 3, 5, 6 (unit); Task 7, 8 (integration); Task 11 (manual E2E) |
| §8 落地次序 15 步 | Reduced to 11 tasks by merging closely related steps; no skipping |
| §9 范围外 / 未来扩展 | Documented in Task 11 README "What this does NOT do" |
| 附录 corpus 选章 | Task 2 fetches the 5 chapters listed |

No gaps.

**2. Placeholder scan:**

- No "TBD" / "implement later" / "add error handling" / "similar to Task N".
- Task 2 step 1 says "fetch from Wikisource and clean wiki markup" — this is a content task, not a code task; the spec provides URLs and explicit cleaning rules. Acceptable.
- Task 11 step 1 is a manual checklist for human verification — explicit pass/fail steps, not vague.
- One stretch: Task 7 step 5 says "If issues: the gateway can return additional canned responses to satisfy the FSM" — this anticipates a possible failure mode without prescribing a fix. Acceptable because the failure is empirically unlikely for the stub agent path (no tools issued by stub gateway).

**3. Type consistency:**

- `BroadcastingEventStore` signature: `constructor(inner: IEventStore)` + `subscribe(contextId, cb) → unsubscribe()`. Defined Task 5; consumed Tasks 7, 8.
- `ConversationSummary` shape `{ contextId, agentId, startedAt, status, runIds, eventCount }`. Defined Task 6; consumed Tasks 7 (GET /conversations response) and 10 (picker render).
- `scanConversations(runsDir): Promise<ConversationSummary[]>` and `readEventsForContext(runsDir, contextId): Promise<Event[]>` — defined Task 6, called Task 7 + Task 8.
- `makeCorpusToolDefinitions(corpusRoot): ToolDefinition[]` — defined Task 3, called Task 7.
- `startServer(config: ServerConfig): Promise<Server>` and `stopServer(server: Server): Promise<void>` — defined Task 7, used Tasks 7 + 8 tests.
- `ServerConfig` fields `{ port, exampleDir, gateway?, agentFile, corpusRoot }` — same shape throughout.
- Frontend JS uses event field names that match server: `event.id`, `event.type`, `event.runId`, `event.timestamp`, `event.payload`. Consistent.
