import { promises as fs } from 'fs'
import path from 'path'
import type { ToolContext } from '../../../src/types/tool.js'

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

  // #37: read a file (optionally a line range) and mint a `passage` object for
  // exactly what was returned. The objectId is the citable handle — the agent
  // cites it later via cite(objectId), so provenance no longer parses prose.
  async function read_file(input: unknown, ctx?: ToolContext): Promise<unknown> {
    const { relPath, lineStart, lineEnd } = input as { relPath: string; lineStart?: number; lineEnd?: number }
    const abs = resolveInsideRoot(relPath)
    const full = await fs.readFile(abs, 'utf-8')
    // Strip a single trailing newline so a file ending in "\n" doesn't report a
    // phantom extra line; 1-based line numbers then match grep's view.
    const allLines = full.replace(/\n$/, '').split('\n')
    const start = typeof lineStart === 'number' && lineStart >= 1 ? lineStart : 1
    const end   = typeof lineEnd === 'number' && lineEnd >= start ? Math.min(lineEnd, allLines.length) : allLines.length
    const content = allLines.slice(start - 1, end).join('\n')
    const obj = ctx?.createObject?.({ type: 'passage', meta: { file: relPath, lineStart: start, lineEnd: end } })
    // objectId + locator FIRST so they survive the truncate(2000) result strategy
    // even when `content` is a long whole-file read — the agent must see objectId to cite.
    return { ...(obj ? { objectId: obj.objectId } : {}), lineStart: start, lineEnd: end, lines: allLines.length, content }
  }

  async function grep(input: unknown, ctx?: ToolContext): Promise<unknown> {
    const { pattern, caseInsensitive = false } = input as { pattern: string; caseInsensitive?: boolean }
    const maxMatches = 50
    const flags = caseInsensitive ? 'gi' : 'g'
    const re = new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), flags)
    const matches: Array<{ file: string; line: number; text: string; objectId?: string }> = []

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
              const file = path.relative(root, abs)
              // #113 P2: grep is wide recall — register each hit lazily (no event)
              // so dozens of never-cited candidates don't flood the log; cite promotes
              // only the one the agent actually uses.
              const obj = ctx?.registerObject?.({ type: 'passage', meta: { file, lineStart: i + 1, lineEnd: i + 1 } })
              matches.push({
                file,
                line: i + 1,
                text: lines[i]!.slice(0, 200),
                ...(obj ? { objectId: obj.objectId } : {}),
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

  // #113 P3: `cite` is now a framework built-in (src/tools/lineage.ts), available
  // to any agent. The corpus tools only produce objects (read_file/grep); the
  // lineage-declaration tools are framework-level.
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
      description: 'Read a file within the corpus, optionally a line range. Returns { objectId, lineStart, lineEnd, lines, content }. Read a tight line range, then pass the returned objectId to the cite tool to source a claim — never write "(chapter:line)" in prose.',
      inputSchema: {
        type: 'object',
        properties: {
          relPath:   { type: 'string', description: 'Path relative to corpus root.' },
          lineStart: { type: 'number', description: 'Optional 1-based first line (inclusive). Omit to read the whole file.' },
          lineEnd:   { type: 'number', description: 'Optional 1-based last line (inclusive).' },
        },
        required: ['relPath'],
      },
      handler: t.read_file,
      // PR-E: chapter bodies are 5K-15K chars; truncating to 2000 keeps tool_result
      // from blowing prefix cache (measured: 75% → 17% hit rate without this).
      // Agent should grep first to locate; read_file gives first 2000 chars of
      // context. For deeper passages, agent can grep again with tighter pattern.
      resultStrategy: { shape: { kind: 'truncate' as const, maxChars: 2000, tailHint: true } },
    },
    {
      name:        'grep',
      description: 'Search for a pattern across all files in the corpus. Returns up to 50 matches, each { file, line, text, objectId }. Pass a match objectId to the cite tool to source a claim from that line.',
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
