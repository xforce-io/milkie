/**
 * Child-process worker for File finalization store races / crash-safe restart (#227 I1).
 *
 * Usage:
 *   npx tsx src/__tests__/helpers/finalizationFileWorker.ts \
 *     create|get|write-crash-safe|read-crash-safe <args...>
 *
 * Prints a single JSON line to stdout and exits 0 on success, 1 on failure.
 */
import { promises as fs } from 'fs'
import path from 'path'
import {
  FileTaskOutcomeFinalizationStore,
} from '../../outcome/TaskOutcomeFinalizationStore.js'
import {
  assembleFinalization,
  buildRecordWithoutHash,
  normalizeFinalizationIntent,
  parseAndValidateFinalization,
} from '../../outcome/finalizationHash.js'
import { JsonlEventStore } from '../../trace/JsonlEventStore.js'
import { FileTraceObjectStore } from '../../trace/TraceObjectStore.js'
import { contentAddressForCanonicalBytes } from '../../trace/hash.js'
import type { TaskOutcomeFinalization } from '../../types/outcome.js'

function emit(payload: unknown, outPath?: string): Promise<void> {
  const line = JSON.stringify(payload)
  process.stdout.write(line + '\n')
  if (outPath) return fs.writeFile(outPath, line, 'utf-8')
  return Promise.resolve()
}

async function main(): Promise<void> {
  const [mode, ...rest] = process.argv.slice(2)
  if (!mode) {
    throw new Error(
      'usage: finalizationFileWorker create|get|write-crash-safe|read-crash-safe ...',
    )
  }

  if (mode === 'create') {
    const [baseDir, arg3, outPath] = rest
    if (!baseDir || !arg3) {
      throw new Error('usage: finalizationFileWorker create <baseDir> <recordPath> [outPath]')
    }
    const store = new FileTaskOutcomeFinalizationStore(baseDir)
    const text = await fs.readFile(arg3, 'utf-8')
    const record = JSON.parse(text) as TaskOutcomeFinalization
    const result = await store.create(record)
    const payload = result.created
      ? { ok: true as const, created: true as const, record: result.record }
      : { ok: true as const, created: false as const, existing: result.existing }
    await emit(payload, outPath)
    return
  }

  if (mode === 'get') {
    const [baseDir, runId, outPath] = rest
    if (!baseDir || !runId) {
      throw new Error('usage: finalizationFileWorker get <baseDir> <runId> [outPath]')
    }
    const store = new FileTaskOutcomeFinalizationStore(baseDir)
    const got = await store.get(runId)
    await emit({ ok: true as const, record: got }, outPath)
    return
  }

  if (mode === 'write-crash-safe') {
    // write-crash-safe <rootDir> <fixtureJsonPath> [outPath]
    const [rootDir, fixturePath, outPath] = rest
    if (!rootDir || !fixturePath) {
      throw new Error(
        'usage: finalizationFileWorker write-crash-safe <rootDir> <fixtureJsonPath> [outPath]',
      )
    }
    const fixture = JSON.parse(await fs.readFile(fixturePath, 'utf-8')) as {
      runId: string
      finalizationId: string
      objectBytes: string
      note?: string
    }

    const eventsDir = path.join(rootDir, 'events')
    const objectsDir = path.join(rootDir, 'objects')
    const finalsDir = path.join(rootDir, 'finals')

    const eventStore = new JsonlEventStore(eventsDir)
    const objectStore = new FileTraceObjectStore(objectsDir)
    const finalStore = new FileTaskOutcomeFinalizationStore(finalsDir)

    const startedId = `${fixture.runId}-started`
    const completedId = `${fixture.runId}-completed`
    const objectId = `${fixture.runId}-obj`

    await eventStore.append({
      id: startedId,
      runId: fixture.runId,
      type: 'agent.run.started',
      actor: 'worker',
      timestamp: 1,
      payload: {},
    })
    await eventStore.append({
      id: completedId,
      runId: fixture.runId,
      type: 'agent.run.completed',
      actor: 'worker',
      timestamp: 2,
      payload: { status: 'completed' },
    })

    const objectHash = await objectStore.putCanonical(fixture.objectBytes) as `sha256:${string}`
    await eventStore.append({
      id: `${fixture.runId}-object-created`,
      runId: fixture.runId,
      type: 'object.created',
      actor: 'worker',
      timestamp: 3,
      payload: { objectId, hash: objectHash },
    })

    // Crash-safe evidence confirmation before final create.
    await eventStore.confirmRunDurable(fixture.runId)
    await objectStore.confirmObjectsDurable([objectHash])

    const { intent, intentHash } = normalizeFinalizationIntent({
      runId: fixture.runId,
      value: 'success',
      verifierClaim: { type: 'eval', id: 'crash-safe-worker' },
      evidence: [
        { kind: 'event', eventId: completedId },
        { kind: 'object', objectId, hash: objectHash },
      ],
      ...(fixture.note !== undefined ? { note: fixture.note } : {}),
    })
    const record = assembleFinalization(buildRecordWithoutHash({
      intent,
      finalizationId: fixture.finalizationId,
      intentHash,
      finalizedAt: 1_700_000_000_111,
    }))

    const created = await finalStore.create(record)
    if (!created.created) {
      throw new Error('write-crash-safe expected created:true')
    }

    await emit({
      ok: true as const,
      runId: fixture.runId,
      objectHash,
      objectBytes: fixture.objectBytes,
      recordHash: created.record.recordHash,
      intentHash: created.record.intentHash,
      finalizationId: created.record.finalizationId,
      completedEventId: completedId,
      objectId,
    }, outPath)
    return
  }

  if (mode === 'read-crash-safe') {
    // read-crash-safe <rootDir> <expectJsonPath> [outPath]
    const [rootDir, expectPath, outPath] = rest
    if (!rootDir || !expectPath) {
      throw new Error(
        'usage: finalizationFileWorker read-crash-safe <rootDir> <expectJsonPath> [outPath]',
      )
    }
    const expected = JSON.parse(await fs.readFile(expectPath, 'utf-8')) as {
      runId: string
      objectHash: string
      objectBytes: string
      recordHash: string
      intentHash: string
      finalizationId: string
      completedEventId: string
      objectId: string
    }

    const eventStore = new JsonlEventStore(path.join(rootDir, 'events'))
    const objectStore = new FileTraceObjectStore(path.join(rootDir, 'objects'))
    const finalStore = new FileTaskOutcomeFinalizationStore(path.join(rootDir, 'finals'))

    const events = await eventStore.readByRunId(expected.runId)
    if (events.length < 2) {
      throw new Error(`expected run events, got ${events.length}`)
    }
    const completed = events.find((e) => e.id === expected.completedEventId)
    if (!completed || completed.type !== 'agent.run.completed') {
      throw new Error('completed event missing after restart')
    }

    const bytes = await objectStore.getCanonical(expected.objectHash)
    if (bytes === undefined) {
      throw new Error('object bytes missing after restart')
    }
    if (bytes !== expected.objectBytes) {
      throw new Error('object bytes mismatch after restart')
    }
    const recomputedObjectHash = contentAddressForCanonicalBytes(bytes)
    if (recomputedObjectHash !== expected.objectHash) {
      throw new Error('object hash recompute mismatch after restart')
    }

    const final = await finalStore.get(expected.runId)
    if (!final) {
      throw new Error('final missing after restart')
    }
    // Re-validate schema/hash strictly in the reader process.
    const validated = parseAndValidateFinalization(final, expected.runId)
    if (validated.recordHash !== expected.recordHash) {
      throw new Error('final recordHash mismatch after restart')
    }
    if (validated.intentHash !== expected.intentHash) {
      throw new Error('final intentHash mismatch after restart')
    }
    if (validated.finalizationId !== expected.finalizationId) {
      throw new Error('finalizationId mismatch after restart')
    }

    await emit({
      ok: true as const,
      events: events.length,
      objectHash: recomputedObjectHash,
      recordHash: validated.recordHash,
      intentHash: validated.intentHash,
      finalizationId: validated.finalizationId,
      value: validated.value,
    }, outPath)
    return
  }

  throw new Error(`unknown mode: ${mode}`)
}

main().catch(async (err) => {
  const payload = {
    ok: false as const,
    name: err?.name ?? 'Error',
    message: err?.message ?? String(err),
    kind: err?.kind,
    stage: err?.stage,
  }
  process.stdout.write(JSON.stringify(payload) + '\n')
  process.stderr.write(String(err?.stack ?? err) + '\n')
  process.exitCode = 1
})
