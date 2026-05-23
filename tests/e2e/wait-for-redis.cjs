const { Redis } = require('ioredis')

async function waitForRedis() {
  const deadline = Date.now() + 30_000
  let lastError

  while (Date.now() < deadline) {
    const client = new Redis({
      host: 'localhost',
      port: 6379,
      db: 15,
      lazyConnect: true,
      maxRetriesPerRequest: 1,
      enableOfflineQueue: false,
      connectTimeout: 500,
      retryStrategy: () => null,
    })
    client.on('error', () => undefined)

    try {
      await client.connect()
      const pong = await client.ping()
      client.disconnect()
      if (pong === 'PONG') return
    } catch (err) {
      lastError = err
      client.disconnect()
      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }

  throw lastError ?? new Error('Redis did not become ready')
}

waitForRedis().catch(err => {
  console.error(err)
  process.exit(1)
})
