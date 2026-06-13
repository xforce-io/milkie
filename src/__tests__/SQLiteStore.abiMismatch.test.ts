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

describe('SQLiteStore — better-sqlite3 ABI mismatch (#157)', () => {
  it('re-throws with a message that names the ABI conflict', async () => {
    const store = new SQLiteStore({ path: ':memory:' })
    await expect(store.init()).rejects.toThrow(/ABI mismatch|NODE_MODULE_VERSION/)
  })

  it('re-throws with the built and required MODULE_VERSION numbers', async () => {
    const store = new SQLiteStore({ path: ':memory:' })
    await expect(store.init()).rejects.toThrow(/131/)
    await expect(store.init()).rejects.toThrow(/127/)
  })

  it('re-throws with an actionable fix command', async () => {
    const store = new SQLiteStore({ path: ':memory:' })
    await expect(store.init()).rejects.toThrow(/npm rebuild better-sqlite3/)
  })
})
