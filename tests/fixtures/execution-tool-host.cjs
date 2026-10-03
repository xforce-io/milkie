// Host process that registers one tool and never answers it. Used to prove host death stops the CLI.
const { ExecutionClient } = require('../../dist/execution/ExecutionClient.js')
let data = ''
process.stdin.setEncoding('utf8')
process.stdin.on('data', chunk => { data += chunk })
process.stdin.on('end', () => {
  try {
    const request = JSON.parse(data)
    const client = new ExecutionClient({ dataDir: request.dataDir, connection: request.connection })
    const runId = client.start(request.contextId, request.input, request.constraints, () => {
      if (request.effect) require('node:fs').writeFileSync(request.effect, 'once')
      return new Promise(() => {})
    })
    process.stdout.write(`${runId}\n`)
    setInterval(() => {}, 1000)
  } catch (error) {
    process.stdout.write(`${error.code ?? 'host_error'}\n`)
    process.exitCode = 1
  }
})
