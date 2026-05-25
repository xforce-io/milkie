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
    const lines = content.trimEnd().split('\n').length
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
