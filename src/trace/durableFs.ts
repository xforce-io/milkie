/**
 * Crash-safe directory helpers for File final / event / object stores (#227).
 *
 * POSIX fsync on a directory only persists entries *inside* that directory.
 * Newly created intermediate directories therefore require fsync of each parent
 * that received a new child entry — leaf → base, and the base parent when the
 * store itself created the configured base root.
 */

import { promises as fs, constants as fsConstants } from 'fs'
import path from 'path'

export type DurableFsStat = (dirPath: string) => Promise<unknown>
export type DurableFsMkdir = (
  dirPath: string,
  opts?: { recursive?: boolean },
) => Promise<string | undefined>
export type DurableFsSyncDirectory = (dirPath: string) => Promise<void>

export interface DurableFsOps {
  mkdir?: DurableFsMkdir
  stat?: DurableFsStat
  syncDirectory?: DurableFsSyncDirectory
}

/** Open a directory O_RDONLY, fsync, close. */
export async function fsyncDirectory(dirPath: string): Promise<void> {
  const fh = await fs.open(dirPath, fsConstants.O_RDONLY)
  try {
    await fh.sync()
  } finally {
    await fh.close()
  }
}

/**
 * Fsync every directory from `startDir` up through `stopAtInclusive` (inclusive),
 * deepest-first. Both paths are resolved; `startDir` must equal stop or lie under it.
 */
export async function fsyncDirectoryChain(
  startDir: string,
  stopAtInclusive: string,
  syncDirectory: DurableFsSyncDirectory = fsyncDirectory,
): Promise<void> {
  const stop = path.resolve(stopAtInclusive)
  let cur = path.resolve(startDir)
  if (cur !== stop && !cur.startsWith(stop + path.sep)) {
    throw new Error(
      `fsyncDirectoryChain: start ${cur} is not within stop ${stop}`,
    )
  }
  for (;;) {
    await syncDirectory(cur)
    if (cur === stop) return
    const parent = path.dirname(cur)
    if (parent === cur) {
      throw new Error(
        `fsyncDirectoryChain: reached filesystem root before stop ${stop}`,
      )
    }
    cur = parent
  }
}

/**
 * mkdir -p with durability of each newly created directory entry.
 *
 * For every directory D that this call creates, fsync(parent(D)) so D's entry
 * is durable before returning. Existing `dirPath` is a no-op.
 *
 * @returns resolved paths of directories this call created (shallow → deep)
 */
export async function mkdirDurable(
  dirPath: string,
  ops: DurableFsOps = {},
): Promise<{ readonly createdDirs: readonly string[] }> {
  const mkdir = ops.mkdir ?? ((p, o) => fs.mkdir(p, o))
  const stat = ops.stat ?? ((p) => fs.stat(p))
  const syncDirectory = ops.syncDirectory ?? fsyncDirectory

  const resolved = path.resolve(dirPath)
  try {
    await stat(resolved)
    return { createdDirs: [] }
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err
  }

  // Walk up until an existing ancestor is found; collect missing dirs deep→shallow.
  const missing: string[] = []
  let cur = resolved
  for (;;) {
    missing.push(cur)
    const parent = path.dirname(cur)
    if (parent === cur) break
    try {
      await stat(parent)
      break
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err
      cur = parent
    }
  }

  // Create shallowest first.
  missing.reverse()
  const createdDirs: string[] = []
  for (const dir of missing) {
    try {
      await mkdir(dir, { recursive: false })
      createdDirs.push(dir)
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err
      // Lost a race; directory now exists — still ensure parent entry is durable below.
    }
    // Persist the new (or raced) child entry in its parent.
    await syncDirectory(path.dirname(dir))
  }
  return { createdDirs }
}

/**
 * After a file link/write under `leafDir`, make the store hierarchy durable:
 * fsync leafDir → … → baseDir (inclusive), then always fsync the parent of
 * the configured baseDir so the base root directory entry itself is durable.
 *
 * Always acknowledging the base parent is required for crash-safe cross-process
 * takeover: a later instance must not skip base-parent fsync merely because it
 * did not create baseDir in this process.
 */
export async function fsyncStoreHierarchy(options: {
  readonly leafDir: string
  readonly baseDir: string
  readonly syncDirectory?: DurableFsSyncDirectory
}): Promise<void> {
  const syncDirectory = options.syncDirectory ?? fsyncDirectory
  const base = path.resolve(options.baseDir)
  const leaf = path.resolve(options.leafDir)
  await fsyncDirectoryChain(leaf, base, syncDirectory)
  const parent = path.dirname(base)
  if (parent !== base) {
    await syncDirectory(parent)
  }
}

/** True when `createdDirs` includes the resolved store base root. */
export function createdIncludesBase(
  createdDirs: readonly string[],
  baseDir: string,
): boolean {
  const base = path.resolve(baseDir)
  return createdDirs.some((d) => path.resolve(d) === base)
}
