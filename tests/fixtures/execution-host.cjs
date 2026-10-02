// A fresh host process for each live SDK turn. Input travels on stdin, never argv.
const { ExecutionClient } = require('../../dist/execution/ExecutionClient.js')
let data = ''
process.stdin.setEncoding('utf8')
process.stdin.on('data', chunk => data += chunk)
process.stdin.on('end', async () => {
  try {
    const request = JSON.parse(data)
    const client = new ExecutionClient({ dataDir: request.dataDir, connection: request.connection })
    const context = request.contextId ? client.getContext(request.contextId) : client.createContext(request.cwd, request.storage)
    const runId = client.start(context.contextId, request.input, request.constraints)
    process.stdout.write(JSON.stringify({ type: 'started', contextId: context.contextId, runId }) + '\n')
    const result = await client.wait(runId, 130000)
    process.stdout.write(JSON.stringify({ type: 'result', result }) + '\n')
  } catch (e) {
    process.stdout.write(JSON.stringify({ type: 'error', code: e.code ?? 'host_error' }) + '\n')
    process.exitCode = 1
  }
})
