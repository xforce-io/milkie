#!/usr/bin/env node
import { EntityResolver, LookupInput, CommitInput } from '../resolver/EntityResolver'

const [, , schemaPath, dataPath] = process.argv
if (!schemaPath || !dataPath) {
  process.stderr.write(
    JSON.stringify({ error: 'Usage: resolver.ts <schemaPath> <dataPath>' }) + '\n',
  )
  process.exit(1)
}

let resolver: EntityResolver
try {
  resolver = new EntityResolver({ schemaPath, dataPath })
} catch (err) {
  process.stderr.write(JSON.stringify({ error: `Failed to load resolver: ${String(err)}` }) + '\n')
  process.exit(1)
}

let raw = ''
process.stdin.setEncoding('utf-8')
process.stdin.on('data', chunk => { raw += chunk })
process.stdin.on('end', () => {
  try {
    const input = JSON.parse(raw) as LookupInput | CommitInput
    let result: unknown
    if (input.op === 'lookup') {
      result = resolver.lookup(input)
    } else if (input.op === 'commit') {
      result = resolver.commit(input)
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
