import { readFileSync } from 'node:fs'
import { isAbsolute } from 'node:path'
import { ExecutionClient } from './ExecutionClient.js'
import { serveExecutionSidecar } from './sidecar.js'
import type { ConnectionInput } from '../connection/types.js'

/** No environment/config auto-discovery: the host supplies the exact connection file. */
export async function runExecutionSidecarCli(args: string[]): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  try {
    if (args.length === 1 && args[0] === '--help') return { stdout: 'Usage: milkie execution-sidecar --data-dir <absolute-path> --connection-file <absolute-path>\n', stderr: '', exitCode: 0 }
    const dataDir = args[1], connectionFile = args[3]
    if (!dataDir || !connectionFile || args.length !== 4 || args[0] !== '--data-dir' || args[2] !== '--connection-file' || !isAbsolute(dataDir) || !isAbsolute(connectionFile)) throw new Error()
    const connection = JSON.parse(readFileSync(connectionFile, 'utf8')) as ConnectionInput
    const client = new ExecutionClient({ dataDir, connection, lifetime: 'host' })
    const controller = new AbortController()
    const stop = () => controller.abort()
    process.once('SIGTERM', stop); process.once('SIGINT', stop)
    try { await serveExecutionSidecar(process.stdin, process.stdout, client, controller.signal) }
    finally { process.off('SIGTERM', stop); process.off('SIGINT', stop) }
    return { stdout: '', stderr: '', exitCode: 0 }
  } catch {
    return { stdout: '', stderr: '{"error":{"code":"SIDECAR_START_FAILED"}}\n', exitCode: 1 }
  }
}
