import { RedisStore } from '../../src/store/RedisStore.js'

export async function createRedisStore(db = 15): Promise<RedisStore> {
  const store = new RedisStore({
    db,
    lazyConnect:          true,
    maxRetriesPerRequest: 1,
    enableOfflineQueue:   false,
    retryStrategy:        () => null,
  })
  try {
    await store.init()
    await store.flushdb()
    return store
  } catch (err) {
    await store.disconnect().catch(() => undefined)
    const reason = err instanceof Error ? err.message : String(err)
    throw new Error(
      `Redis e2e requires a reachable Redis server on localhost:6379 (db ${db}). ` +
      `Start Redis or run npm run test:e2e:redis. Original error: ${reason}`,
    )
  }
}
