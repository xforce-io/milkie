import { once } from 'events'
import * as fs from 'fs'
import * as http from 'http'
import * as path from 'path'
import {
  collectFromPrefix,
  resolveAndParseConnection,
  assembleApiGateway,
  ConnectionConfigError,
  createGateway,
  type ConnectionInput,
  type ConnectionParseResult,
} from '../index'
import { LoggingGateway } from '../logging/LoggingGateway'

const PREFIX = 'HELIX_LLM_'
const SECRET = 'sk-test'
const BASE = 'https://example.invalid/v1'
const BASE2 = 'https://example.invalid/openai'

function expectNoSecrets(value: unknown): void {
  const serialized = JSON.stringify(value)
  expect(serialized).not.toContain(SECRET)
  expect(serialized).not.toContain(BASE)
  expect(serialized).not.toContain(BASE2)
}
function expectPublicAssembly(value: { adapterFamily: string; gateway: unknown }): void {
  expect(value.gateway).toBeInstanceOf(LoggingGateway)
  expect(Object.keys(value).sort()).toEqual(['adapterFamily', 'gateway'])
}

function header(req: http.IncomingMessage, name: string): string | undefined {
  const value = req.headers[name]
  return typeof value === 'string' ? value : undefined
}

async function listenJsonStub(
  onRequest: (req: http.IncomingMessage, res: http.ServerResponse) => void,
): Promise<{ origin: string; close: () => Promise<void> }> {
  const server = http.createServer((req, res) => {
    res.statusCode = 200
    res.setHeader('content-type', 'application/json')
    onRequest(req, res)
  })
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const address = server.address()
  if (!address || typeof address === 'string') throw new Error('stub has no TCP address')
  return {
    origin: `http://127.0.0.1:${address.port}`,
    close: async () => {
      server.close()
      await once(server, 'close')
    },
  }
}

function expectError(
  fn: () => unknown,
  code: string,
  fields: string[],
): ConnectionConfigError {
  try {
    fn()
    throw new Error(`expected ${code}`)
  } catch (err) {
    expect(err).toBeInstanceOf(ConnectionConfigError)
    const e = err as ConnectionConfigError
    expect(e.code).toBe(code)
    expect(e.fields).toEqual(fields)
    expectNoSecrets({ code: e.code, message: e.message, fields: e.fields })
    return e
  }
}

describe('collectFromPrefix', () => {
  it('treats a missing suffix as absent, not empty string', () => {
    const fields = collectFromPrefix(PREFIX, { HELIX_LLM_TRANSPORT: 'api' })
    expect(fields).toEqual({ transport: 'api' })
    expect(Object.prototype.hasOwnProperty.call(fields, 'apiKey')).toBe(false)
    expect(fields.apiKey).toBeUndefined()
  })

  it('does not read vendor variables or process.env', () => {
    const prev = process.env['VOLCENGINE_TOKEN']
    process.env['VOLCENGINE_TOKEN'] = 'sk-process'
    process.env['ANTHROPIC_API_KEY'] = 'sk-anth'
    try {
      const fields = collectFromPrefix(PREFIX, {
        HELIX_LLM_MODEL: 'm',
        ANTHROPIC_API_KEY: 'sk-anth',
        VOLCENGINE_TOKEN: 'sk-volc',
      })
      expect(fields).toEqual({ model: 'm' })
    } finally {
      if (prev === undefined) delete process.env['VOLCENGINE_TOKEN']
      else process.env['VOLCENGINE_TOKEN'] = prev
      delete process.env['ANTHROPIC_API_KEY']
    }
  })
})

describe('resolveAndParseConnection — S1 canonical api', () => {
  it('parses anthropic-messages and keeps secrets off the projection', () => {
    const result = resolveAndParseConnection({
      contractVersion: 1,
      fields: {
        transport: 'api',
        protocol: 'anthropic-messages',
        model: 'claude-x',
        apiKey: SECRET,
        baseUrl: BASE,
        provider: 'ignored-for-routing',
      },
    })
    expect(result.projection).toEqual({
      contractVersion: 1,
      transport: 'api',
      protocol: 'anthropic-messages',
      model: 'claude-x',
      provider: 'ignored-for-routing',
      hasApiKey: true,
      hasBaseUrl: true,
      source: 'canonical',
    })
    expectNoSecrets(result)
    const assembled = assembleApiGateway(result)
    expect(assembled.adapterFamily).toBe('anthropic')
    expectPublicAssembly(assembled)
  })

  it('parses openai-chat-completions; provider does not change the adapter family', () => {
    const a = resolveAndParseConnection({
      contractVersion: 1,
      fields: {
        transport: 'api',
        protocol: 'openai-chat-completions',
        model: 'gpt-x',
        apiKey: SECRET,
        baseUrl: BASE2,
        provider: 'volcengine',
      },
    })
    const b = resolveAndParseConnection({
      contractVersion: 1,
      fields: {
        transport: 'api',
        protocol: 'openai-chat-completions',
        model: 'gpt-x',
        apiKey: SECRET,
        baseUrl: BASE2,
        provider: 'openai',
      },
    })
    expect(a.projection.protocol).toBe('openai-chat-completions')
    expect(assembleApiGateway(a).adapterFamily).toBe('openai-compatible')
    expect(assembleApiGateway(b).adapterFamily).toBe('openai-compatible')
  })
})

describe('resolveAndParseConnection — S2 transport split', () => {
  it('accepts agent-cli without creating a gateway', () => {
    const result = resolveAndParseConnection({
      contractVersion: 1,
      fields: { transport: 'agent-cli', runtime: 'claude-code', model: 'opus' },
    })
    expect(result.projection).toMatchObject({
      transport: 'agent-cli',
      runtime: 'claude-code',
      model: 'opus',
      hasApiKey: false,
      hasBaseUrl: false,
      source: 'canonical',
    })
    expect(() => assembleApiGateway(result)).toThrow(ConnectionConfigError)
  })

  it('rejects api+runtime and agent-cli+protocol before any call', () => {
    expectError(() => resolveAndParseConnection({
      contractVersion: 1,
      fields: {
        transport: 'api',
        protocol: 'anthropic-messages',
        model: 'm',
        apiKey: SECRET,
        runtime: 'claude-code',
      },
    }), 'CONNECTION_CONFIG_CONFLICT', ['runtime'])

    expectError(() => resolveAndParseConnection({
      contractVersion: 1,
      fields: {
        transport: 'agent-cli',
        runtime: 'grok-cli',
        protocol: 'anthropic-messages',
      },
    }), 'CONNECTION_CONFIG_CONFLICT', ['protocol'])
  })
})

describe('resolveAndParseConnection — entry A matches entry B', () => {
  it('produces the same projection for prefix snapshot and hand-built fields', () => {
    const fromB = resolveAndParseConnection({
      contractVersion: 1,
      fields: {
        transport: 'api',
        protocol: 'anthropic-messages',
        model: 'claude-x',
        apiKey: SECRET,
      },
    })
    const fromA = resolveAndParseConnection({
      contractVersion: 1,
      prefix: PREFIX,
      env: {
        HELIX_LLM_TRANSPORT: 'api',
        HELIX_LLM_PROTOCOL: 'anthropic-messages',
        HELIX_LLM_MODEL: 'claude-x',
        HELIX_LLM_API_KEY: SECRET,
      },
    })
    expect(fromA.projection).toEqual(fromB.projection)
  })
})

describe('resolveAndParseConnection — validation', () => {
  it('rejects missing contractVersion and unknown extra keys', () => {
    expectError(() => resolveAndParseConnection({
      fields: { transport: 'api', protocol: 'anthropic-messages', model: 'm', apiKey: SECRET },
    }), 'CONNECTION_CONFIG_MISSING_FIELD', ['contractVersion'])

    expectError(() => resolveAndParseConnection({
      contractVersion: 1,
      fields: {
        transport: 'api',
        protocol: 'anthropic-messages',
        model: 'm',
        apiKey: SECRET,
        extra: 'x',
        also: 'y',
      },
    }), 'CONNECTION_CONFIG_UNKNOWN_VALUE', ['also', 'extra'])
  })

  it('rejects blank values and does not trim', () => {
    expectError(() => resolveAndParseConnection({
      contractVersion: 1,
      fields: {
        transport: 'api',
        protocol: 'anthropic-messages',
        model: ' m',
        apiKey: SECRET,
      },
    }), 'CONNECTION_CONFIG_UNKNOWN_VALUE', ['model'])

    const kept = resolveAndParseConnection({
      contractVersion: 1,
      fields: {
        transport: 'api',
        protocol: 'openai-chat-completions',
        model: 'my model',
        apiKey: SECRET,
        baseUrl: 'https://example.invalid/path with space',
      },
    })
    expect(kept.projection.model).toBe('my model')
    expect(assembleApiGateway(kept).adapterFamily).toBe('openai-compatible')
  })

  it('reports mixed canonical+legacy before blank-value errors', () => {
    expectError(() => resolveAndParseConnection({
      contractVersion: 1,
      prefix: PREFIX,
      env: {
        HELIX_LLM_API_KEY: '  ',
        ANTHROPIC_API_KEY: SECRET,
      },
    }), 'CONNECTION_CONFIG_LEGACY_AND_CANONICAL', ['ANTHROPIC_API_KEY', 'apiKey'])
  })
})

describe('resolveAndParseConnection — S4 legacy', () => {
  const legacyModelConfig = { adapter: 'anthropic', model: 'claude-x' }

  it('maps in-window legacy ModelConfig plus vendor key', () => {
    const fromA = resolveAndParseConnection({
      contractVersion: 1,
      prefix: PREFIX,
      env: { ANTHROPIC_API_KEY: SECRET, VOLCENGINE_API_BASE: BASE },
      legacyModelConfig,
    })
    const fromB = resolveAndParseConnection({
      contractVersion: 1,
      fields: {},
      legacyModelConfig,
      legacyEnv: { ANTHROPIC_API_KEY: SECRET, VOLCENGINE_API_BASE: BASE },
    })
    expect(fromA.projection).toEqual(fromB.projection)
    expect(fromA.projection).toMatchObject({
      transport: 'api',
      protocol: 'anthropic-messages',
      model: 'claude-x',
      hasApiKey: true,
      hasBaseUrl: true,
      source: 'legacy',
    })
    expect(assembleApiGateway(fromA).adapterFamily).toBe('anthropic')
    expectNoSecrets(fromA)
  })

  it('rejects only-legacy input outside the window', () => {
    expectError(() => resolveAndParseConnection({
      contractVersion: 2,
      fields: {},
      legacyModelConfig,
      legacyEnv: { ANTHROPIC_API_KEY: SECRET },
    }), 'CONNECTION_CONFIG_LEGACY_EXPIRED', ['ANTHROPIC_API_KEY', 'legacyModelConfig'])
  })

  it('treats vendor key alone as legacy source that still lacks fields', () => {
    expectError(() => resolveAndParseConnection({
      contractVersion: 1,
      fields: {},
      legacyEnv: { ANTHROPIC_API_KEY: SECRET },
    }), 'CONNECTION_CONFIG_MISSING_FIELD', ['transport'])

    expectError(() => resolveAndParseConnection({
      contractVersion: 2,
      fields: {},
      legacyEnv: { ANTHROPIC_API_KEY: SECRET },
    }), 'CONNECTION_CONFIG_LEGACY_EXPIRED', ['ANTHROPIC_API_KEY'])
  })
})

describe('assembleApiGateway — S1 wiring', () => {
  it('builds an Anthropic family gateway without exposing secrets on the public object', () => {
    const parsed = resolveAndParseConnection({
      contractVersion: 1,
      fields: {
        transport: 'api',
        protocol: 'anthropic-messages',
        model: 'claude-x',
        apiKey: SECRET,
        baseUrl: BASE,
      },
    })
    const assembled = assembleApiGateway(parsed)
    expect(assembled.adapterFamily).toBe('anthropic')
    expectPublicAssembly(assembled)
  })

  it('sends a real HTTP request to the configured Anthropic base URL', async () => {
    const seen: Array<{ url?: string; apiKey?: string }> = []
    const stub = await listenJsonStub((req, res) => {
      seen.push({ url: req.url, apiKey: header(req, 'x-api-key') })
      res.end(JSON.stringify({
        id: 'msg_1',
        type: 'message',
        role: 'assistant',
        model: 'claude-x',
        content: [{ type: 'text', text: 'ok' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 1, output_tokens: 1 },
      }))
    })
    try {
      const assembled = assembleApiGateway(resolveAndParseConnection({
        contractVersion: 1,
        fields: {
          transport: 'api',
          protocol: 'anthropic-messages',
          model: 'claude-x',
          apiKey: SECRET,
          baseUrl: stub.origin,
        },
      }))
      const response = await assembled.gateway.complete({
        model: 'claude-x',
        messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      })
      expect(response.content[0]).toEqual({ type: 'text', text: 'ok' })
      expect(seen).toHaveLength(1)
      expect(seen[0]?.url).toContain('messages')
      expect(seen[0]?.apiKey).toBe(SECRET)
    } finally {
      await stub.close()
    }
  })

  it('builds an OpenAI-compatible family gateway for the other protocol', () => {
    const parsed = resolveAndParseConnection({
      contractVersion: 1,
      fields: {
        transport: 'api',
        protocol: 'openai-chat-completions',
        model: 'gpt-x',
        apiKey: SECRET,
        baseUrl: BASE2,
      },
    })
    const assembled = assembleApiGateway(parsed)
    expect(assembled.adapterFamily).toBe('openai-compatible')
    expectPublicAssembly(assembled)
  })

  it('sends a real HTTP request to the configured OpenAI-compatible base URL', async () => {
    const seen: Array<{ url?: string; authorization?: string }> = []
    const stub = await listenJsonStub((req, res) => {
      seen.push({ url: req.url, authorization: header(req, 'authorization') })
      res.end(JSON.stringify({
        id: 'chatcmpl-1',
        choices: [{
          index: 0,
          message: { role: 'assistant', content: 'ok' },
          finish_reason: 'stop',
        }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      }))
    })
    try {
      const assembled = assembleApiGateway(resolveAndParseConnection({
        contractVersion: 1,
        fields: {
          transport: 'api',
          protocol: 'openai-chat-completions',
          model: 'gpt-x',
          apiKey: SECRET,
          baseUrl: `${stub.origin}/v1`,
        },
      }))
      const response = await assembled.gateway.complete({
        model: 'gpt-x',
        messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }],
      })
      expect(response.content[0]).toEqual({ type: 'text', text: 'ok' })
      expect(seen).toHaveLength(1)
      expect(seen[0]?.url).toContain('chat/completions')
      expect(seen[0]?.authorization).toBe(`Bearer ${SECRET}`)
    } finally {
      await stub.close()
    }
  })

  it('leaves the old createGateway path unchanged', () => {
    const gw = createGateway({ provider: 'anthropic', model: 'claude-x', adapter: 'anthropic' })
    expect(gw).toBeInstanceOf(LoggingGateway)
  })
})

describe('conformance fixtures', () => {
  const root = path.join(__dirname, '../../contracts/model-connection')

  function loadCases(versionDir: string): Array<Record<string, unknown>> {
    const dir = path.join(root, versionDir)
    return fs.readdirSync(dir)
      .filter(name => name.endsWith('.json'))
      .map(name => JSON.parse(fs.readFileSync(path.join(dir, name), 'utf8')) as Record<string, unknown>)
  }

  function kindsOf(cases: Array<Record<string, unknown>>): Record<string, true> {
    const kinds: Record<string, true> = {}
    for (const item of cases) kinds[String(item.kind)] = true
    return kinds
  }

  it('ships at least four fixture kinds per contract version', () => {
    expect(kindsOf(loadCases('v1'))).toMatchObject({
      success: true,
      conflict: true,
      missing: true,
      redaction: true,
    })
    expect(kindsOf(loadCases('v2'))).toMatchObject({
      success: true,
      conflict: true,
      missing: true,
      redaction: true,
    })
  })

  it('evaluates every fixture against the Node reference', () => {
    for (const version of ['v1', 'v2']) {
      for (const fixture of loadCases(version)) {
        const run = (): ConnectionParseResult =>
          resolveAndParseConnection(fixture.input as ConnectionInput)
        if (fixture.kind === 'success' || fixture.kind === 'redaction') {
          const result = run()
          expect(result.projection).toEqual(fixture.expect)
          expectNoSecrets(result)
          expectNoSecrets(fixture.expect)
        } else {
          const expected = fixture.expect as { code: string; fields: string[] }
          expectError(run, expected.code, expected.fields)
        }
      }
    }
  })
})
