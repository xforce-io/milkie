// #157: When better-sqlite3's native .node file was compiled for a different
// Node version, the import throws a raw error with "NODE_MODULE_VERSION" in the
// message. SQLiteStore.init() must catch it and re-throw a human-readable error
// that names the mismatch and the fix command, instead of letting the generic
// CLI_ERROR wrapper swallow the root cause.
jest.mock('better-sqlite3', () => {
  throw new Error(
    "The module '/path/to/better_sqlite3.node' was compiled against a different Node.js version " +
    "using NODE_MODULE_VERSION 131. " +
    "This version of Node.js requires NODE_MODULE_VERSION 127. " +
    "Please try re-compiling or re-installing the module (for instance, using `npm rebuild` or `npm install`)."
  )
})

import { SQLiteStore } from '../store/SQLiteStore'

// The raw better-sqlite3 error opens with this — used as the negative control so
// each assertion proves init() produced a NEW message rather than letting the raw
// error pass through (which would also contain "NODE_MODULE_VERSION", "131", "127").
const RAW_PREFIX = "The module '"

describe('SQLiteStore — better-sqlite3 ABI mismatch (#157)', () => {
  it('re-throws a NEW "ABI mismatch" message, not the raw module error', async () => {
    const store = new SQLiteStore({ path: ':memory:' })
    const err = (await store.init().catch((e: unknown) => e)) as Error
    expect(err).toBeInstanceOf(Error)
    expect(err.message).toMatch(/^better-sqlite3 ABI mismatch:/)
    // discriminating: a bare rethrow would start with the raw "The module '…'"
    expect(err.message).not.toContain(RAW_PREFIX)
  })

  it('names the built and required MODULE_VERSION in the rewritten phrasing', async () => {
    const store = new SQLiteStore({ path: ':memory:' })
    const err = (await store.init().catch((e: unknown) => e)) as Error
    // "built for"/"requires …" is the new message's wording; the raw error says
    // "using NODE_MODULE_VERSION 131", so these only pass when init() rewrote it.
    expect(err.message).toContain('built for NODE_MODULE_VERSION 131')
    expect(err.message).toContain('requires NODE_MODULE_VERSION 127')
    expect(err.message).not.toContain(RAW_PREFIX)
  })

  it('re-throws with an actionable fix command', async () => {
    const store = new SQLiteStore({ path: ':memory:' })
    await expect(store.init()).rejects.toThrow(/npm rebuild better-sqlite3/)
  })
})
