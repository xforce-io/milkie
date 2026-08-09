/**
 * Unit tests for #227 / s-017 task outcome finalization.
 * Covers hash stability, evidence validation, memory races, file atomic create,
 * idempotency matrix, and observation isolation.
 */

import { promises as fs, constants as fsConstants } from 'fs'
import { createHash } from 'crypto'
import { spawn } from 'child_process'
import os from 'os'
import path from 'path'
import { Milkie } from '../runtime/Milkie'
import { MemoryStore } from '../store/MemoryStore'
import { MemoryEventStore } from '../trace/MemoryEventStore'
import { MemoryTraceObjectStore, FileTraceObjectStore } from '../trace/TraceObjectStore'
import { JsonlEventStore } from '../trace/JsonlEventStore'
import {
  fsyncDirectoryChain,
  fsyncStoreHierarchy,
  mkdirDurable,
} from '../trace/durableFs'
import { contentAddressForCanonicalBytes } from '../trace/hash'
import type { Event } from '../trace/types'
import type { IModelGateway, ModelRequest, ModelResponse } from '../types/model'
import type { AgentConfig } from '../types/agent'
import type {
  EvidenceRef,
  FinalizeTaskOutcomeInput,
  TaskOutcomeFinalization,
} from '../types/outcome'
import {
  TaskOutcomeEvidenceError,
  TaskOutcomeFinalizationConfigurationError,
  TaskOutcomeFinalizationCorruptionError,
  TaskOutcomeFinalizationStoreError,
  TaskOutcomeFinalizationValidationError,
  TaskOutcomeRunNotFoundError,
} from '../types/outcome'
import {
  assembleFinalization,
  buildRecordWithoutHash,
  normalizeFinalizationIntent,
  parseAndValidateFinalization,
} from '../outcome/finalizationHash'
import { resolveAgainstExisting } from '../outcome/validateEvidence'
import {
  FileTaskOutcomeFinalizationStore,
  MemoryTaskOutcomeFinalizationStore,
  type FileFinalizationFsOps,
} from '../outcome/TaskOutcomeFinalizationStore'

class FixedGateway implements IModelGateway {
  public callCount = 0
  constructor(private readonly text: string) {}
  async complete(_req: ModelRequest): Promise<ModelResponse> {
    this.callCount++
    return {
      content: [{ type: 'text', text: this.text }],
      toolCalls: [],
      finishReason: 'end_turn',
    }
  }
  async *stream(_req: ModelRequest): AsyncIterable<never> {
    yield* []
  }
}

const agent: AgentConfig = {
  agentId: 'finalization-unit-agent',
  version: '0.0.0',
  systemPrompt: 'answer briefly',
  fsm: {
    states: [
      { name: 'reply', type: 'llm', terminal: true, tools: [] },
    ],
  },
  model: { provider: 'stub', model: 'stub', adapter: 'stub' },
} as AgentConfig

function baseIntent(overrides: Partial<{
  runId: string
  value: TaskOutcomeFinalization['value']
  eventId: string
  note: string
  scores: { name: string; value: number | string | boolean }[]
}> = {}) {
  const eventId = overrides.eventId ?? 'ev-1'
  return {
    runId: overrides.runId ?? 'run-1',
    value: overrides.value ?? ('success' as const),
    verifierClaim: { type: 'eval' as const, id: 'suite-a' },
    evidence: [{ kind: 'event' as const, eventId }],
    ...(overrides.note !== undefined ? { note: overrides.note } : {}),
    ...(overrides.scores !== undefined ? { scores: overrides.scores } : {}),
  }
}

function makeRecord(overrides: {
  runId?: string
  finalizationId?: string
  value?: TaskOutcomeFinalization['value']
  eventId?: string
  finalizedAt?: number
  note?: string
} = {}): TaskOutcomeFinalization {
  const { intent, intentHash } = normalizeFinalizationIntent(baseIntent(overrides))
  const without = buildRecordWithoutHash({
    intent,
    finalizationId: overrides.finalizationId ?? 'fid-1',
    intentHash,
    finalizedAt: overrides.finalizedAt ?? 1_700_000_000_000,
  })
  return assembleFinalization(without)
}

async function seedCompletedRun(
  eventStore: MemoryEventStore,
  runId: string,
  extra: Event[] = [],
): Promise<{ completedEventId: string; startedEventId: string }> {
  const startedEventId = `${runId}-started`
  const completedEventId = `${runId}-completed`
  await eventStore.append({
    id: startedEventId,
    runId,
    type: 'agent.run.started',
    actor: 'test',
    timestamp: 1,
    payload: { agentId: 'a', goal: 'g', input: 'i', contextId: 'c' },
  })
  for (const e of extra) await eventStore.append(e)
  await eventStore.append({
    id: completedEventId,
    runId,
    type: 'agent.run.completed',
    actor: 'test',
    timestamp: 2,
    payload: { status: 'completed' },
  })
  return { completedEventId, startedEventId }
}

describe('finalizationHash', () => {
  it('produces stable intentHash independent of evidence input order', () => {
    const a = normalizeFinalizationIntent({
      runId: ' r1 ',
      value: 'success',
      verifierClaim: { type: 'eval', id: ' v1 ' },
      evidence: [
        { kind: 'object', objectId: 'o2', hash: 'sha256:' + 'b'.repeat(64) },
        { kind: 'event', eventId: 'e1' },
        { kind: 'object', objectId: 'o1', hash: 'sha256:' + 'a'.repeat(64) },
      ],
      scores: [
        { name: ' b ', value: 2 },
        { name: 'a', value: true },
      ],
    })
    const b = normalizeFinalizationIntent({
      runId: 'r1',
      value: 'success',
      verifierClaim: { type: 'eval', id: 'v1' },
      evidence: [
        { kind: 'event', eventId: 'e1' },
        { kind: 'object', objectId: 'o1', hash: 'sha256:' + 'a'.repeat(64) },
        { kind: 'object', objectId: 'o2', hash: 'sha256:' + 'b'.repeat(64) },
      ],
      scores: [
        { name: 'a', value: true },
        { name: 'b', value: 2 },
      ],
    })
    expect(a.intentHash).toBe(b.intentHash)
    expect(a.intentHash).toMatch(/^sha256:[a-f0-9]{64}$/)
    expect(a.intent.evidence.map(e => e.kind + (e.kind === 'event' ? e.eventId : e.objectId))).toEqual([
      'evente1',
      'objecto1',
      'objecto2',
    ])
    expect(a.intent.scores?.map(s => s.name)).toEqual(['a', 'b'])
  })

  it('excludes finalizationId and finalizedAt from intentHash; includes them in recordHash', () => {
    const { intent, intentHash } = normalizeFinalizationIntent(baseIntent())
    const r1 = assembleFinalization(buildRecordWithoutHash({
      intent, finalizationId: 'fid-a', intentHash, finalizedAt: 100,
    }))
    const r2 = assembleFinalization(buildRecordWithoutHash({
      intent, finalizationId: 'fid-b', intentHash, finalizedAt: 200,
    }))
    expect(r1.intentHash).toBe(r2.intentHash)
    expect(r1.recordHash).not.toBe(r2.recordHash)
    expect(r1.recordHash).toMatch(/^sha256:[a-f0-9]{64}$/)
  })

  it('rejects duplicate evidence, empty evidence, bad claim, and JSON-unsafe values', () => {
    expect(() => normalizeFinalizationIntent({
      ...baseIntent(),
      evidence: [],
    })).toThrow(TaskOutcomeFinalizationValidationError)

    expect(() => normalizeFinalizationIntent({
      ...baseIntent(),
      evidence: [
        { kind: 'event', eventId: 'e1' },
        { kind: 'event', eventId: 'e1' },
      ],
    })).toThrow(/duplicate/)

    expect(() => normalizeFinalizationIntent({
      ...baseIntent(),
      verifierClaim: { type: 'robot', id: 'x' },
    })).toThrow(TaskOutcomeFinalizationValidationError)

    expect(() => normalizeFinalizationIntent({
      ...baseIntent(),
      scores: [{ name: 's', value: { nested: 1 } as unknown as number }],
    })).toThrow(TaskOutcomeFinalizationValidationError)
  })

  it('detects recordHash mismatch as validation failure', () => {
    const record = makeRecord()
    const tampered = { ...record, value: 'failure' as const }
    expect(() => parseAndValidateFinalization(tampered, record.runId)).toThrow(
      /recordHash mismatch|intentHash/,
    )
  })

  it('rejects unknown top-level fields even when recordHash matches known fields', () => {
    const record = makeRecord()
    // Valid hash for known fields, plus an extra top-level key that must fail closed.
    const withExtra = { ...record, extraAdminNote: 'should-not-be-accepted' }
    expect(() => parseAndValidateFinalization(withExtra, record.runId)).toThrow(
      /unknown top-level field/,
    )
    expect(() => parseAndValidateFinalization(withExtra, record.runId)).toThrow(
      TaskOutcomeFinalizationValidationError,
    )
  })

  it('File store maps unknown top-level fields on disk to corruption', async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'milkie-final-extra-'))
    try {
      const store = new FileTaskOutcomeFinalizationStore(dir)
      const record = makeRecord({ runId: 'run-extra-field' })
      await store.create(record)
      const hex = createHash('sha256').update('run-extra-field').digest('hex')
      const target = path.join(dir, 'sha256', hex.slice(0, 2), `${hex.slice(2)}.json`)
      const onDisk = JSON.parse(await fs.readFile(target, 'utf-8')) as Record<string, unknown>
      onDisk.extraAdminNote = 'tamper'
      await fs.writeFile(target, JSON.stringify(onDisk), 'utf-8')
      await expect(store.get('run-extra-field')).rejects.toMatchObject({
        name: 'TaskOutcomeFinalizationCorruptionError',
        code: 'task_outcome_finalization_corruption',
      })
    } finally {
      await fs.rm(dir, { recursive: true, force: true })
    }
  })
})

describe('idempotency matrix (resolveAgainstExisting)', () => {
  const existing = makeRecord({ finalizationId: 'fid-1', value: 'success' })

  it('same finalizationId + same intentHash → idempotent', () => {
    const r = resolveAgainstExisting(existing, {
      finalizationId: 'fid-1',
      value: 'success',
      intentHash: existing.intentHash,
    })
    expect(r.status).toBe('idempotent')
    if (r.status === 'idempotent') expect(r.final.recordHash).toBe(existing.recordHash)
  })

  it('same finalizationId + different intentHash → idempotency_key_reused', () => {
    const other = makeRecord({ finalizationId: 'fid-1', value: 'failure', eventId: 'other' })
    const r = resolveAgainstExisting(existing, {
      finalizationId: 'fid-1',
      value: 'failure',
      intentHash: other.intentHash,
    })
    expect(r.status).toBe('conflict')
    if (r.status === 'conflict') {
      expect(r.conflict.kind).toBe('idempotency_key_reused')
      expect(r.conflict.attempted).toEqual({
        finalizationId: 'fid-1',
        value: 'failure',
        intentHash: other.intentHash,
      })
      expect('note' in r.conflict.attempted).toBe(false)
      expect('evidence' in r.conflict.attempted).toBe(false)
    }
  })

  it('different finalizationId → already_finalized even with same value', () => {
    const r = resolveAgainstExisting(existing, {
      finalizationId: 'fid-2',
      value: 'success',
      intentHash: existing.intentHash,
    })
    expect(r.status).toBe('conflict')
    if (r.status === 'conflict') expect(r.conflict.kind).toBe('already_finalized')
  })
})

describe('MemoryTaskOutcomeFinalizationStore', () => {
  it('create-if-absent; concurrent creates yield one winner', async () => {
    const store = new MemoryTaskOutcomeFinalizationStore()
    const a = makeRecord({ finalizationId: 'a', finalizedAt: 1 })
    const b = makeRecord({ finalizationId: 'b', finalizedAt: 2, value: 'failure', eventId: 'ev-x' })

    const results = await Promise.all([
      store.create(a),
      store.create(b),
      store.create(a),
    ])
    const created = results.filter(r => r.created)
    const existing = results.filter(r => !r.created)
    expect(created).toHaveLength(1)
    expect(existing).toHaveLength(2)

    const winnerHash = created[0]!.record.recordHash
    for (const r of existing) {
      expect(r.existing.recordHash).toBe(winnerHash)
    }
    const got = await store.get(a.runId)
    expect(got!.recordHash).toBe(winnerHash)
  })

  it('returns independent snapshots; mutating input/result does not corrupt store', async () => {
    const store = new MemoryTaskOutcomeFinalizationStore()
    const record = makeRecord()
    const mutableEvidence = [...record.evidence] as EvidenceRef[]
    const input = { ...record, evidence: mutableEvidence }

    const created = await store.create(input)
    expect(created.created).toBe(true)
    if (!created.created) return

    // Mutate input after create
    mutableEvidence.push({ kind: 'event', eventId: 'injected' })
    ;(created.record as { value: string }).value = 'failure'

    const got = await store.get(record.runId)
    expect(got!.value).toBe('success')
    expect(got!.evidence).toHaveLength(1)
    expect(got!.recordHash).toBe(record.recordHash)

    // Mutating get result does not affect next get
    ;(got!.evidence as EvidenceRef[]).push({ kind: 'event', eventId: 'x' })
    const got2 = await store.get(record.runId)
    expect(got2!.evidence).toHaveLength(1)
  })
})

describe('FileTaskOutcomeFinalizationStore', () => {
  let dir: string

  beforeEach(async () => {
    dir = await fs.mkdtemp(path.join(os.tmpdir(), 'milkie-final-'))
  })
  afterEach(async () => {
    await fs.rm(dir, { recursive: true, force: true })
  })

  it('atomic create-if-absent; second create returns existing; survives reopen', async () => {
    const store = new FileTaskOutcomeFinalizationStore(dir)
    const record = makeRecord({ runId: 'run-file-1' })
    const r1 = await store.create(record)
    expect(r1.created).toBe(true)

    const other = makeRecord({ runId: 'run-file-1', finalizationId: 'other', value: 'failure', eventId: 'e9' })
    const r2 = await store.create(other)
    expect(r2.created).toBe(false)
    if (!r2.created) {
      expect(r2.existing.recordHash).toBe(record.recordHash)
      expect(r2.existing.finalizationId).toBe('fid-1')
    }

    const reopened = new FileTaskOutcomeFinalizationStore(dir)
    const got = await reopened.get('run-file-1')
    expect(got!.recordHash).toBe(record.recordHash)
    expect(got!.value).toBe('success')
  })

  it('directory hierarchy fsync failure after link yields commit_unknown and does not return final', async () => {
    let afterLink = false
    const real = {
      mkdir: (p: string, opts?: { recursive?: boolean }) => fs.mkdir(p, opts),
      open: async (p: string, flags: string | number, mode?: number) => {
        const fh = await fs.open(p, flags, mode)
        return {
          writeFile: (data: string, encoding: BufferEncoding) => fh.writeFile(data, encoding),
          sync: () => fh.sync(),
          close: () => fh.close(),
        }
      },
      link: async (a: string, b: string) => {
        await fs.link(a, b)
        afterLink = true
      },
      readFile: (p: string, enc: BufferEncoding) => fs.readFile(p, enc),
      rm: (p: string, opts?: { force?: boolean }) => fs.rm(p, opts),
      syncDirectory: async (dirPath: string) => {
        // Allow durable mkdir path; fail only on post-link hierarchy confirmation.
        if (afterLink) throw new Error('injected hierarchy dir sync failure')
        const fh = await fs.open(dirPath, fsConstants.O_RDONLY)
        try { await fh.sync() } finally { await fh.close() }
      },
      readdir: (p: string) => fs.readdir(p),
      stat: (p: string) => fs.stat(p),
    } satisfies FileFinalizationFsOps

    const store = new FileTaskOutcomeFinalizationStore(dir, { fsOps: real })
    const record = makeRecord({ runId: 'run-fsync-fail' })
    await expect(store.create(record)).rejects.toMatchObject({
      name: 'TaskOutcomeFinalizationStoreError',
      kind: 'commit_unknown',
    })

    // Target may exist on disk but get must also fsync hierarchy; with working sync it returns.
    const store2 = new FileTaskOutcomeFinalizationStore(dir)
    const got = await store2.get('run-fsync-fail')
    // After a successful get-side hierarchy fsync, the record becomes visible.
    expect(got?.recordHash).toBe(record.recordHash)
  })

  it('parent-dir fsync fault injection across newly-created levels yields commit_unknown', async () => {
    // Nested base so mkdirDurable creates base itself and must fsync base parent.
    const nestedBase = path.join(dir, 'nested', 'final-root')
    const syncedAfterLink: string[] = []
    let afterLink = false

    const real = {
      mkdir: (p: string, opts?: { recursive?: boolean }) => fs.mkdir(p, opts),
      open: async (p: string, flags: string | number, mode?: number) => {
        const fh = await fs.open(p, flags, mode)
        return {
          writeFile: (data: string, encoding: BufferEncoding) => fh.writeFile(data, encoding),
          sync: () => fh.sync(),
          close: () => fh.close(),
        }
      },
      link: async (a: string, b: string) => {
        await fs.link(a, b)
        afterLink = true
      },
      readFile: (p: string, enc: BufferEncoding) => fs.readFile(p, enc),
      rm: (p: string, opts?: { force?: boolean }) => fs.rm(p, opts),
      syncDirectory: async (dirPath: string) => {
        const resolved = path.resolve(dirPath)
        if (afterLink) {
          syncedAfterLink.push(resolved)
          // Let the leaf shard succeed, then fail a higher parent so leaf-only
          // fsync would incorrectly acknowledge while hierarchy remains undurable.
          const baseResolved = path.resolve(nestedBase)
          if (resolved === baseResolved || resolved === path.resolve(path.dirname(baseResolved))) {
            throw new Error(`injected parent fsync failure for ${resolved}`)
          }
        }
        const fh = await fs.open(dirPath, fsConstants.O_RDONLY)
        try { await fh.sync() } finally { await fh.close() }
      },
      readdir: (p: string) => fs.readdir(p),
      stat: (p: string) => fs.stat(p),
    } satisfies FileFinalizationFsOps

    const store = new FileTaskOutcomeFinalizationStore(nestedBase, { fsOps: real })
    const record = makeRecord({ runId: 'run-parent-levels' })
    await expect(store.create(record)).rejects.toMatchObject({
      name: 'TaskOutcomeFinalizationStoreError',
      kind: 'commit_unknown',
      stage: 'create_winner_dir_fsync',
    })

    // Hierarchy confirmation must walk above the leaf shard.
    expect(syncedAfterLink.length).toBeGreaterThanOrEqual(2)
    const hex = createHash('sha256').update('run-parent-levels').digest('hex')
    const leafShard = path.resolve(nestedBase, 'sha256', hex.slice(0, 2))
    expect(syncedAfterLink[0]).toBe(leafShard)
    expect(
      syncedAfterLink.some(
        (p) => p === path.resolve(nestedBase) || p === path.resolve(path.dirname(nestedBase)),
      ),
    ).toBe(true)

    // Recovery reader with healthy fsync sees the linked target.
    const recovered = new FileTaskOutcomeFinalizationStore(nestedBase)
    const got = await recovered.get('run-parent-levels')
    expect(got?.recordHash).toBe(record.recordHash)
  })

  it('two child processes race create on shared File store; one winner, loser gets same existing', async () => {
    const recordA = makeRecord({ runId: 'run-xproc', finalizationId: 'fid-a', value: 'success', eventId: 'e-a' })
    const recordB = makeRecord({ runId: 'run-xproc', finalizationId: 'fid-b', value: 'failure', eventId: 'e-b' })
    const aPath = path.join(dir, 'record-a.json')
    const bPath = path.join(dir, 'record-b.json')
    await fs.writeFile(aPath, JSON.stringify(recordA), 'utf-8')
    await fs.writeFile(bPath, JSON.stringify(recordB), 'utf-8')

    const worker = path.join(__dirname, 'helpers', 'finalizationFileWorker.ts')
    const repoRoot = path.resolve(__dirname, '..', '..')
    const spawnCreate = (recordPath: string) =>
      new Promise<{ code: number | null; stdout: string; stderr: string }>((resolve) => {
        const child = spawn('npx', ['tsx', worker, 'create', dir, recordPath], {
          cwd: repoRoot,
          env: { ...process.env, FORCE_COLOR: '0' },
          stdio: ['ignore', 'pipe', 'pipe'],
        })
        let stdout = ''
        let stderr = ''
        child.stdout.on('data', (b: Buffer) => { stdout += b.toString() })
        child.stderr.on('data', (b: Buffer) => { stderr += b.toString() })
        child.on('close', (code) => resolve({ code, stdout, stderr }))
      })

    const [ra, rb] = await Promise.all([spawnCreate(aPath), spawnCreate(bPath)])
    expect(ra.code).toBe(0)
    expect(rb.code).toBe(0)

    const parseLine = (stdout: string) =>
      JSON.parse(stdout.trim().split('\n').pop()!) as {
        ok: boolean
        created: boolean
        record?: TaskOutcomeFinalization
        existing?: TaskOutcomeFinalization
        message?: string
      }

    const pa = parseLine(ra.stdout)
    const pb = parseLine(rb.stdout)
    expect(pa.ok).toBe(true)
    expect(pb.ok).toBe(true)

    const createdCount = Number(pa.created) + Number(pb.created)
    expect(createdCount).toBe(1)

    const winnerHash = pa.created ? pa.record!.recordHash : pa.existing!.recordHash
    const otherHash = pb.created ? pb.record!.recordHash : pb.existing!.recordHash
    expect(otherHash).toBe(winnerHash)

    // Exactly one target file under sha256/
    const shaRoot = path.join(dir, 'sha256')
    const shards = await fs.readdir(shaRoot)
    let jsonCount = 0
    for (const shard of shards) {
      const files = await fs.readdir(path.join(shaRoot, shard))
      jsonCount += files.filter((f) => f.endsWith('.json')).length
    }
    expect(jsonCount).toBe(1)
  }, 60000)

  it('writer child exits after crash-safe write; reader child rebuilds three stores and rechecks hashes', async () => {
    const root = path.join(dir, 'xproc-restart-root')
    await fs.mkdir(root, { recursive: true })
    const fixturePath = path.join(dir, 'write-fixture.json')
    const expectPath = path.join(dir, 'write-expect.json')
    const fixture = {
      runId: 'run-xproc-restart',
      finalizationId: 'fid-xproc-restart',
      objectBytes: 'canonical evidence body for real process restart',
      note: 'writer-child-exit',
    }
    await fs.writeFile(fixturePath, JSON.stringify(fixture), 'utf-8')

    const worker = path.join(__dirname, 'helpers', 'finalizationFileWorker.ts')
    const repoRoot = path.resolve(__dirname, '..', '..')
    const runWorker = (args: string[]) =>
      new Promise<{ code: number | null; stdout: string; stderr: string }>((resolve) => {
        const child = spawn('npx', ['tsx', worker, ...args], {
          cwd: repoRoot,
          env: { ...process.env, FORCE_COLOR: '0' },
          stdio: ['ignore', 'pipe', 'pipe'],
        })
        let stdout = ''
        let stderr = ''
        child.stdout.on('data', (b: Buffer) => { stdout += b.toString() })
        child.stderr.on('data', (b: Buffer) => { stderr += b.toString() })
        child.on('close', (code) => resolve({ code, stdout, stderr }))
      })

    const written = await runWorker(['write-crash-safe', root, fixturePath])
    expect(written.code).toBe(0)
    const writePayload = JSON.parse(written.stdout.trim().split('\n').pop()!) as {
      ok: boolean
      runId: string
      objectHash: string
      objectBytes: string
      recordHash: string
      intentHash: string
      finalizationId: string
      completedEventId: string
      objectId: string
    }
    expect(writePayload.ok).toBe(true)
    expect(writePayload.recordHash).toMatch(/^sha256:[a-f0-9]{64}$/)
    await fs.writeFile(expectPath, JSON.stringify(writePayload), 'utf-8')

    // Writer process has fully exited before the independent reader starts.
    const read = await runWorker(['read-crash-safe', root, expectPath])
    expect(read.code).toBe(0)
    const readPayload = JSON.parse(read.stdout.trim().split('\n').pop()!) as {
      ok: boolean
      objectHash: string
      recordHash: string
      intentHash: string
      finalizationId: string
      value: string
    }
    expect(readPayload.ok).toBe(true)
    expect(readPayload.objectHash).toBe(writePayload.objectHash)
    expect(readPayload.recordHash).toBe(writePayload.recordHash)
    expect(readPayload.intentHash).toBe(writePayload.intentHash)
    expect(readPayload.finalizationId).toBe(writePayload.finalizationId)
    expect(readPayload.value).toBe('success')
  }, 90000)

  it('takeover instance always fsyncs configured base parent even when it did not create base', async () => {
    // Process A creates baseDir under nested parent, then a second store instance
    // (simulating takeover after A exits) must still fsync base parent on get/create ack.
    const nestedBase = path.join(dir, 'takeover', 'final-root')
    await fs.mkdir(path.dirname(nestedBase), { recursive: true })

    const creator = new FileTaskOutcomeFinalizationStore(nestedBase)
    const record = makeRecord({ runId: 'run-takeover-base-parent' })
    const created = await creator.create(record)
    expect(created.created).toBe(true)

    const synced: string[] = []
    const real = {
      mkdir: (p: string, opts?: { recursive?: boolean }) => fs.mkdir(p, opts),
      open: async (p: string, flags: string | number, mode?: number) => {
        const fh = await fs.open(p, flags, mode)
        return {
          writeFile: (data: string, encoding: BufferEncoding) => fh.writeFile(data, encoding),
          sync: () => fh.sync(),
          close: () => fh.close(),
        }
      },
      link: (a: string, b: string) => fs.link(a, b),
      readFile: (p: string, enc: BufferEncoding) => fs.readFile(p, enc),
      rm: (p: string, opts?: { force?: boolean }) => fs.rm(p, opts),
      syncDirectory: async (dirPath: string) => {
        synced.push(path.resolve(dirPath))
        const fh = await fs.open(dirPath, fsConstants.O_RDONLY)
        try { await fh.sync() } finally { await fh.close() }
      },
      readdir: (p: string) => fs.readdir(p),
      stat: (p: string) => fs.stat(p),
    } satisfies FileFinalizationFsOps

    const takeover = new FileTaskOutcomeFinalizationStore(nestedBase, { fsOps: real })
    const got = await takeover.get('run-takeover-base-parent')
    expect(got?.recordHash).toBe(record.recordHash)

    const baseParent = path.resolve(path.dirname(nestedBase))
    expect(synced).toContain(baseParent)
    expect(synced).toContain(path.resolve(nestedBase))
  })

  it('malformed existing fails closed with corruption error', async () => {
    const store = new FileTaskOutcomeFinalizationStore(dir)
    const record = makeRecord({ runId: 'run-corrupt' })
    await store.create(record)

    // Tamper the file bytes
    const hex = createHash('sha256').update('run-corrupt').digest('hex')
    const target = path.join(dir, 'sha256', hex.slice(0, 2), `${hex.slice(2)}.json`)
    await fs.writeFile(target, '{"schemaVersion":1,"state":"finalized","tampered":true}\n', 'utf-8')

    await expect(store.get('run-corrupt')).rejects.toBeInstanceOf(TaskOutcomeFinalizationCorruptionError)
    const other = makeRecord({ runId: 'run-corrupt', finalizationId: 'new' })
    await expect(store.create(other)).rejects.toBeInstanceOf(TaskOutcomeFinalizationCorruptionError)
  })
})

describe('Milkie finalizeTaskOutcome / getFinalTaskOutcome', () => {
  it('S1 path (process): finalize once with event+object evidence; observation isolation', async () => {
    const eventStore = new MemoryEventStore()
    const objectStore = new MemoryTraceObjectStore()
    const finalStore = new MemoryTaskOutcomeFinalizationStore()
    const gateway = new FixedGateway('ok')
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway,
      eventStore,
      traceObjectStore: objectStore,
      outcomeFinalizationStore: finalStore,
    })
    milkie.registerAgent(agent)

    // Seed a completed run with object evidence via direct store writes (unit-level).
    const runId = 'run-s1-unit'
    const bytes = JSON.stringify({ answer: 42 })
    const hash = await objectStore.putCanonical(bytes) as `sha256:${string}`
    const { completedEventId } = await seedCompletedRun(eventStore, runId, [
      {
        id: `${runId}-obj`,
        runId,
        type: 'object.created',
        actor: 'test',
        timestamp: 1.5,
        payload: {
          objectId: 'obj:answer',
          type: 'passage',
          producerEventId: `${runId}-started`,
          hash,
        },
      },
    ])

    const input: FinalizeTaskOutcomeInput = {
      runId,
      expectedState: 'unfinalized',
      finalizationId: 'finalize-1',
      value: 'success',
      verifierClaim: { type: 'eval', id: 'unit-suite' },
      evidence: [
        { kind: 'event', eventId: completedEventId },
        { kind: 'object', objectId: 'obj:answer', hash },
      ],
      note: 'unit s1',
      scores: [{ name: 'score', value: 1 }],
    }

    const result = await milkie.finalizeTaskOutcome(input)
    expect(result.status).toBe('finalized')
    if (result.status !== 'finalized') return

    expect(result.final.state).toBe('finalized')
    expect(result.final.value).toBe('success')
    expect(result.final.verifierClaim).toEqual({ type: 'eval', id: 'unit-suite' })
    expect(result.final.evidence).toHaveLength(2)
    expect(result.final.intentHash).toMatch(/^sha256:/)
    expect(result.final.recordHash).toMatch(/^sha256:/)

    const got = await milkie.getFinalTaskOutcome(runId)
    expect(got!.recordHash).toBe(result.final.recordHash)

    // Idempotent retry
    const again = await milkie.finalizeTaskOutcome(input)
    expect(again.status).toBe('idempotent')
    if (again.status === 'idempotent') {
      expect(again.final.recordHash).toBe(result.final.recordHash)
    }

    // Observation LWW remains independent
    await milkie.recordTaskOutcome({ runId, value: 'failure', source: 'human', note: 'override obs' })
    const obs = await milkie.getTaskOutcome(runId)
    expect(obs!.value).toBe('failure')
    const finalAfterObs = await milkie.getFinalTaskOutcome(runId)
    expect(finalAfterObs!.value).toBe('success')
    expect(finalAfterObs!.recordHash).toBe(result.final.recordHash)

    // No LLM on finalize/query
    expect(gateway.callCount).toBe(0)
  })

  it('S2 path: conflict matrix; never overwrites existing final', async () => {
    const eventStore = new MemoryEventStore()
    const finalStore = new MemoryTaskOutcomeFinalizationStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway: new FixedGateway('x'),
      eventStore,
      outcomeFinalizationStore: finalStore,
    })

    const runId = 'run-s2-unit'
    const { completedEventId } = await seedCompletedRun(eventStore, runId)

    const base: FinalizeTaskOutcomeInput = {
      runId,
      expectedState: 'unfinalized',
      finalizationId: 'winner',
      value: 'success',
      verifierClaim: { type: 'rule', id: 'r1' },
      evidence: [{ kind: 'event', eventId: completedEventId }],
    }

    const w = await milkie.finalizeTaskOutcome(base)
    expect(w.status).toBe('finalized')
    if (w.status !== 'finalized') return
    const winnerHash = w.final.recordHash

    // different finalizationId → already_finalized
    const c1 = await milkie.finalizeTaskOutcome({
      ...base,
      finalizationId: 'loser-1',
      value: 'failure',
    })
    expect(c1.status).toBe('conflict')
    if (c1.status === 'conflict') {
      expect(c1.conflict.kind).toBe('already_finalized')
      expect(c1.existing.recordHash).toBe(winnerHash)
    }

    // same key + different intent → idempotency_key_reused
    const c2 = await milkie.finalizeTaskOutcome({
      ...base,
      value: 'partial',
    })
    expect(c2.status).toBe('conflict')
    if (c2.status === 'conflict') {
      expect(c2.conflict.kind).toBe('idempotency_key_reused')
      expect(c2.existing.recordHash).toBe(winnerHash)
    }

    // same key + same intent → idempotent
    const idemp = await milkie.finalizeTaskOutcome(base)
    expect(idemp.status).toBe('idempotent')

    const final = await milkie.getFinalTaskOutcome(runId)
    expect(final!.recordHash).toBe(winnerHash)
    expect(final!.value).toBe('success')
  })

  it('getFinal null for unfinalized; unknown run throws; missing stores throw config', async () => {
    const eventStore = new MemoryEventStore()
    const finalStore = new MemoryTaskOutcomeFinalizationStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway: new FixedGateway('x'),
      eventStore,
      outcomeFinalizationStore: finalStore,
    })
    const runId = 'run-unfinalized'
    await seedCompletedRun(eventStore, runId)

    await expect(milkie.getFinalTaskOutcome(runId)).resolves.toBeNull()
    await expect(milkie.getFinalTaskOutcome('no-such-run')).rejects.toBeInstanceOf(
      TaskOutcomeRunNotFoundError,
    )

    const noFinal = new Milkie({
      stateStore: new MemoryStore(),
      gateway: new FixedGateway('x'),
      eventStore,
    })
    await expect(
      noFinal.finalizeTaskOutcome({
        runId,
        expectedState: 'unfinalized',
        finalizationId: 'x',
        value: 'success',
        verifierClaim: { type: 'eval', id: 'e' },
        evidence: [{ kind: 'event', eventId: 'e' }],
      }),
    ).rejects.toBeInstanceOf(TaskOutcomeFinalizationConfigurationError)
  })

  it('crash-safe final rejects Memory evidence stores', async () => {
    const eventStore = new MemoryEventStore()
    const finalStore = new FileTaskOutcomeFinalizationStore(
      await fs.mkdtemp(path.join(os.tmpdir(), 'milkie-final-cfg-')),
    )
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway: new FixedGateway('x'),
      eventStore,
      outcomeFinalizationStore: finalStore,
    })
    const runId = 'run-cfg'
    const { completedEventId } = await seedCompletedRun(eventStore, runId)
    await expect(
      milkie.finalizeTaskOutcome({
        runId,
        expectedState: 'unfinalized',
        finalizationId: 'x',
        value: 'success',
        verifierClaim: { type: 'eval', id: 'e' },
        evidence: [{ kind: 'event', eventId: completedEventId }],
      }),
    ).rejects.toBeInstanceOf(TaskOutcomeFinalizationConfigurationError)
  })

  it('rejects missing completed event and bad object evidence', async () => {
    const eventStore = new MemoryEventStore()
    const objectStore = new MemoryTraceObjectStore()
    const finalStore = new MemoryTaskOutcomeFinalizationStore()
    const milkie = new Milkie({
      stateStore: new MemoryStore(),
      gateway: new FixedGateway('x'),
      eventStore,
      traceObjectStore: objectStore,
      outcomeFinalizationStore: finalStore,
    })

    // No completed
    await eventStore.append({
      id: 's',
      runId: 'run-nc',
      type: 'agent.run.started',
      actor: 't',
      timestamp: 1,
      payload: { agentId: 'a', goal: 'g', input: 'i', contextId: 'c' },
    })
    await expect(
      milkie.finalizeTaskOutcome({
        runId: 'run-nc',
        expectedState: 'unfinalized',
        finalizationId: 'x',
        value: 'success',
        verifierClaim: { type: 'eval', id: 'e' },
        evidence: [{ kind: 'event', eventId: 's' }],
      }),
    ).rejects.toBeInstanceOf(TaskOutcomeEvidenceError)

    // Object without hash / bytes
    const runId = 'run-obj-bad'
    const { completedEventId } = await seedCompletedRun(eventStore, runId, [
      {
        id: 'obj-ev',
        runId,
        type: 'object.created',
        actor: 't',
        timestamp: 1.5,
        payload: {
          objectId: 'obj:x',
          type: 'passage',
          producerEventId: `${runId}-started`,
          // no hash
        },
      },
    ])
    const fakeHash = ('sha256:' + 'c'.repeat(64)) as `sha256:${string}`
    await expect(
      milkie.finalizeTaskOutcome({
        runId,
        expectedState: 'unfinalized',
        finalizationId: 'x',
        value: 'success',
        verifierClaim: { type: 'eval', id: 'e' },
        evidence: [
          { kind: 'event', eventId: completedEventId },
          { kind: 'object', objectId: 'obj:x', hash: fakeHash },
        ],
      }),
    ).rejects.toBeInstanceOf(TaskOutcomeEvidenceError)
  })

  it('public error classes are instanceof-checkable', () => {
    expect(new TaskOutcomeFinalizationValidationError('x')).toBeInstanceOf(Error)
    expect(new TaskOutcomeFinalizationConfigurationError('x')).toBeInstanceOf(Error)
    expect(new TaskOutcomeEvidenceError('event_not_found', 'x')).toBeInstanceOf(Error)
    expect(new TaskOutcomeFinalizationStoreError('commit_unknown', 's', 'x')).toBeInstanceOf(Error)
    expect(new TaskOutcomeFinalizationCorruptionError('r', 'x')).toBeInstanceOf(Error)
  })
})

describe('durableFs mkdir + hierarchy helpers', () => {
  let dir: string
  beforeEach(async () => {
    dir = await fs.mkdtemp(path.join(os.tmpdir(), 'milkie-durable-fs-'))
  })
  afterEach(async () => {
    await fs.rm(dir, { recursive: true, force: true })
  })

  it('mkdirDurable fsyncs every parent of newly created dirs', async () => {
    const target = path.join(dir, 'a', 'b', 'c')
    const synced: string[] = []
    const { createdDirs } = await mkdirDurable(target, {
      syncDirectory: async (p) => {
        synced.push(path.resolve(p))
      },
    })
    expect(createdDirs.map((p) => path.resolve(p))).toEqual([
      path.resolve(dir, 'a'),
      path.resolve(dir, 'a', 'b'),
      path.resolve(dir, 'a', 'b', 'c'),
    ])
    // Each created dir's parent is fsynced (dir, a, b).
    expect(synced).toEqual([
      path.resolve(dir),
      path.resolve(dir, 'a'),
      path.resolve(dir, 'a', 'b'),
    ])
  })

  it('fsyncDirectoryChain walks leaf → stop inclusive', async () => {
    const leaf = path.join(dir, 'x', 'y')
    await fs.mkdir(leaf, { recursive: true })
    const order: string[] = []
    await fsyncDirectoryChain(leaf, dir, async (p) => {
      order.push(path.resolve(p))
    })
    expect(order).toEqual([
      path.resolve(dir, 'x', 'y'),
      path.resolve(dir, 'x'),
      path.resolve(dir),
    ])
  })

  it('fsyncStoreHierarchy always fsyncs configured base parent', async () => {
    const base = path.join(dir, 'store-base')
    const leaf = path.join(base, 'sha256', 'ab')
    await fs.mkdir(leaf, { recursive: true })
    const order: string[] = []
    await fsyncStoreHierarchy({
      leafDir: leaf,
      baseDir: base,
      syncDirectory: async (p) => {
        order.push(path.resolve(p))
      },
    })
    expect(order).toEqual([
      path.resolve(leaf),
      path.resolve(base, 'sha256'),
      path.resolve(base),
      path.resolve(dir), // base parent — always, not gated on who created base
    ])
  })
})

describe('JsonlEventStore / FileTraceObjectStore durable confirm hierarchy', () => {
  let dir: string
  beforeEach(async () => {
    dir = await fs.mkdtemp(path.join(os.tmpdir(), 'milkie-ev-obj-'))
  })
  afterEach(async () => {
    await fs.rm(dir, { recursive: true, force: true })
  })

  it('confirmRunDurable survives process restart reader after durable append', async () => {
    const eventsDir = path.join(dir, 'events-root')
    const store = new JsonlEventStore(eventsDir)
    const runId = 'run-jsonl-durable'
    await store.append({
      id: 'e1',
      runId,
      type: 'agent.run.started',
      actor: 't',
      timestamp: 1,
      payload: {},
    })
    await store.confirmRunDurable(runId)

    // New process-equivalent instance (fresh class state) can still read.
    const reopened = new JsonlEventStore(eventsDir)
    const events = await reopened.readByRunId(runId)
    expect(events).toHaveLength(1)
    expect(events[0]!.id).toBe('e1')
  })

  it('confirmObjectsDurable walks leaf→base and survives reopen', async () => {
    const objectsDir = path.join(dir, 'objects-root')
    const store = new FileTraceObjectStore(objectsDir)
    const bytes = JSON.stringify({ k: 'v' })
    const hash = await store.putCanonical(bytes) as `sha256:${string}`
    await store.confirmObjectsDurable([hash])

    const reopened = new FileTraceObjectStore(objectsDir)
    const got = await reopened.getCanonical(hash)
    expect(got).toBe(bytes)
    expect(contentAddressForCanonicalBytes(got!)).toBe(hash)
  })
})

