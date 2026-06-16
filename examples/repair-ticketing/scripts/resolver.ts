#!/usr/bin/env node
//
// Example-scoped CLI wrapper for the portable entity resolver (#167).
// This is NOT a published milkie binary — it is invoked via `node scripts/resolver.ts`.
//
// argv schema:
//   node scripts/resolver.ts <schemaPath> <dataPath>
//     <schemaPath>  path to the column-mapping schema JSON
//     <dataPath>    path to the data CSV interpreted by the schema
//
// This wrapper is the only filesystem-touching layer: it reads the files and
// feeds the fs-free core. The target level is derived from how many ancestors
// are pinned (next unfilled level), so the request never carries an explicit
// level. A single JSON request is read from stdin and a single JSON line is
// written to stdout (errors -> stderr, exit 1):
//   lookup: { "op": "lookup", "utterance": "...", "pinned": { } }
//   commit: { "op": "commit", "selected": "...", "utterance": "...", "pinned": { } }
//
import fs from 'fs'
import {
  loadHierarchicalDict,
  lookupEntities,
  commitEntities,
  type Schema,
  type HierarchicalDict,
} from '../resolver/EntityResolver'

interface LookupRequestJson { op: 'lookup'; utterance: string; pinned?: Record<string, string> }
interface CommitRequestJson { op: 'commit'; selected: string; utterance?: string; pinned?: Record<string, string> }

const [, , schemaPath, dataPath] = process.argv
if (!schemaPath || !dataPath) {
  process.stderr.write(
    JSON.stringify({ error: 'Usage: resolver.ts <schemaPath> <dataPath>' }) + '\n',
  )
  process.exit(1)
}

let dict: HierarchicalDict
try {
  const schema = JSON.parse(fs.readFileSync(schemaPath, 'utf-8')) as Schema
  const csv = fs.readFileSync(dataPath, 'utf-8')
  dict = loadHierarchicalDict(schema, csv)
} catch (err) {
  process.stderr.write(JSON.stringify({ error: `Failed to load resolver: ${String(err)}` }) + '\n')
  process.exit(1)
}

let raw = ''
process.stdin.setEncoding('utf-8')
process.stdin.on('data', chunk => { raw += chunk })
process.stdin.on('end', () => {
  try {
    const input = JSON.parse(raw) as LookupRequestJson | CommitRequestJson
    const pinned = input.pinned ?? {}
    // No explicit level: the core derives the target level from `pinned`.
    let result: unknown
    if (input.op === 'lookup') {
      result = lookupEntities({ utterance: input.utterance, pinned, dict })
    } else if (input.op === 'commit') {
      result = commitEntities({ selected: input.selected, utterance: input.utterance, pinned, dict })
    } else {
      throw new Error(`Unknown op: "${(input as { op: string }).op}"`)
    }
    process.stdout.write(JSON.stringify(result) + '\n')
    process.exit(0)
  } catch (err) {
    process.stderr.write(JSON.stringify({ error: String(err) }) + '\n')
    process.exit(1)
  }
})
