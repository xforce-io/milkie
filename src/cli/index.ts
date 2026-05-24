#!/usr/bin/env node
import { main } from './main.js'

main(process.argv.slice(2)).then((result) => {
  if (result.stdout) process.stdout.write(result.stdout)
  if (result.stderr) process.stderr.write(result.stderr)
  process.exit(result.exitCode)
}).catch((err: unknown) => {
  const message = err instanceof Error ? err.message : String(err)
  process.stderr.write(JSON.stringify({ error: { code: 'FATAL', message } }) + '\n')
  process.exit(1)
})
