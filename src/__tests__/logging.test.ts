import { createServiceLogger, resolveFormat, getLogger, setLogger } from '../logging/logger'

/** 内存 sink：每条 NDJSON 一行，parse 后断言字段。 */
function memorySink(): { lines: () => Record<string, unknown>[]; stream: { write: (s: string) => void } } {
  const raw: string[] = []
  return {
    lines: () => raw.flatMap(s => s.split('\n').filter(Boolean)).map(s => JSON.parse(s) as Record<string, unknown>),
    stream: { write: (s: string) => { raw.push(s) } },
  }
}

describe('createServiceLogger', () => {
  it('emits NDJSON with ts(ISO8601)/level-label/msg', () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    log.info({ port: 8080 }, 'serve listening')
    const line = sink.lines()[0]!
    expect(line.msg).toBe('serve listening')
    expect(line.level).toBe('info')                                   // 字符串标签而非数字
    expect(line.port).toBe(8080)
    expect(typeof line.ts).toBe('string')
    expect(() => new Date(line.ts as string).toISOString()).not.toThrow()
    expect(line.pid).toBeUndefined()                                  // base 字段去掉
    expect(line.hostname).toBeUndefined()
  })

  it('filters below configured level', () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'warn', format: 'json', destination: sink.stream })
    log.info('hidden')
    log.warn('shown')
    expect(sink.lines().map(l => l.msg)).toEqual(['shown'])
  })

  it('level=silent emits nothing', () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'silent', format: 'json', destination: sink.stream })
    log.error('nope')
    expect(sink.lines()).toEqual([])
  })

  it('child loggers merge mod and correlation ids', () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    log.child({ mod: 'server' }).child({ runId: 'r1', contextId: 'c1' }).info('http request')
    const line = sink.lines()[0]!
    expect(line.mod).toBe('server')
    expect(line.runId).toBe('r1')
    expect(line.contextId).toBe('c1')
  })

  it('serializes err with message and stack', () => {
    const sink = memorySink()
    const log = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    log.error({ err: new Error('boom') }, 'llm call failed')
    const line = sink.lines()[0]!
    const err = line.err as { message: string; stack: string }
    expect(err.message).toBe('boom')
    expect(err.stack).toContain('Error: boom')
  })

  it('defaults level from LOG_LEVEL env', () => {
    const prev = process.env['LOG_LEVEL']
    process.env['LOG_LEVEL'] = 'error'
    try {
      const sink = memorySink()
      const log = createServiceLogger({ format: 'json', destination: sink.stream })
      log.warn('hidden')
      log.error('shown')
      expect(sink.lines().map(l => l.msg)).toEqual(['shown'])
    } finally {
      if (prev === undefined) delete process.env['LOG_LEVEL']
      else process.env['LOG_LEVEL'] = prev
    }
  })
})

describe('resolveFormat', () => {
  it('LOG_FORMAT=json overrides TTY', () => expect(resolveFormat({ LOG_FORMAT: 'json' }, true)).toBe('json'))
  it('LOG_FORMAT=pretty overrides non-TTY', () => expect(resolveFormat({ LOG_FORMAT: 'pretty' }, false)).toBe('pretty'))
  it('defaults to pretty on TTY', () => expect(resolveFormat({}, true)).toBe('pretty'))
  it('defaults to json off TTY', () => expect(resolveFormat({}, false)).toBe('json'))
  it('ignores unknown LOG_FORMAT values', () => expect(resolveFormat({ LOG_FORMAT: 'xml' }, false)).toBe('json'))
})

describe('getLogger / setLogger', () => {
  afterEach(() => setLogger(undefined))
  it('getLogger returns a stable lazy singleton', () => {
    expect(getLogger()).toBe(getLogger())
  })
  it('setLogger swaps the singleton (test injection); setLogger(undefined) resets', () => {
    const sink = memorySink()
    const injected = createServiceLogger({ level: 'info', format: 'json', destination: sink.stream })
    setLogger(injected)
    expect(getLogger()).toBe(injected)
    setLogger(undefined)
    expect(getLogger()).not.toBe(injected)
  })
})
