/**
 * #85: standalone launcher for the interrupt/resume sidecar.
 *
 * Run as its own OS process — this is the cross-process counterpart that an
 * external driver (e.g. alfred, or the cross-process e2e test) talks to over
 * HTTP:
 *
 *   PORT=8090 npx tsx examples/interrupt-resume-sidecar/main.ts
 *
 * With PORT unset (or 0) the OS assigns a free port; the chosen port is printed
 * as `SIDECAR_READY <port>` on stdout so a parent process can discover it.
 */
import { createSidecar, buildDemoMilkie } from './sidecar.js'

const demo = buildDemoMilkie({
  totalSteps: process.env['STEPS'] ? Number(process.env['STEPS']) : undefined,
  stepMs:     process.env['STEP_MS'] ? Number(process.env['STEP_MS']) : undefined,
})
const server = createSidecar({
  milkie:   demo.milkie,
  agentId:  demo.agentId,
  progress: () => demo.executedSteps().length,
})

const port = Number(process.env['PORT'] ?? 0)
server.listen(port, () => {
  const addr = server.address()
  const chosen = typeof addr === 'object' && addr ? addr.port : port
  process.stdout.write(`SIDECAR_READY ${chosen}\n`)
})

function shutdown(): void {
  server.close(() => process.exit(0))
  // Force-exit if connections linger past a grace period.
  setTimeout(() => process.exit(0), 1000).unref()
}
process.on('SIGTERM', shutdown)
process.on('SIGINT', shutdown)
