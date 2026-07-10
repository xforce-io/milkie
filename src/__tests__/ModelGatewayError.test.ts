import {
  ModelGatewayError,
  normalizeModelGatewayError,
} from '../gateway/ModelGatewayError'

describe('normalizeModelGatewayError', () => {
  test.each([
    ['APIConnectionError', undefined, 'MODEL_CONNECTION_ERROR', true],
    ['APITimeoutError', undefined, 'MODEL_TIMEOUT', true],
    ['APIError', 429, 'MODEL_RATE_LIMITED', true],
    ['APIError', 401, 'MODEL_AUTH_ERROR', false],
    ['APIError', 403, 'MODEL_AUTH_ERROR', false],
    ['APIError', 400, 'MODEL_BAD_RESPONSE', false],
    ['APIError', 503, 'MODEL_BAD_RESPONSE', true],
  ])('%s status=%s maps to %s', (name, status, code, retryable) => {
    const cause = Object.assign(new Error('provider detail'), { name, status })
    const error = normalizeModelGatewayError(cause, {
      provider: 'volcengine', model: 'glm-5.2', phase: 'stream_open',
    })

    expect(error).toBeInstanceOf(ModelGatewayError)
    expect(error.toJSON()).toMatchObject({
      code, retryable, provider: 'volcengine', model: 'glm-5.2', phase: 'stream_open',
    })
    expect(error.cause).toBe(cause)
  })

  test('classifies a nested transport errno without serializing the cause', () => {
    const cause = Object.assign(new Error('request failed'), {
      cause: Object.assign(new Error('socket'), { code: 'ECONNRESET' }),
    })
    const error = normalizeModelGatewayError(cause, {
      provider: 'volcengine', model: 'glm-5.2', phase: 'stream_read',
    })

    expect(error.toJSON()).toEqual({
      code: 'MODEL_CONNECTION_ERROR',
      message: 'Model provider connection failed.',
      phase: 'stream_read',
      provider: 'volcengine',
      model: 'glm-5.2',
      retryable: true,
    })
    expect(JSON.stringify(error)).not.toContain('ECONNRESET')
  })

  test('recognizes SDK subclasses even when the public error name is generic', () => {
    class APIConnectionError extends Error {}
    const cause = new APIConnectionError('connection detail')
    cause.name = 'Error'
    const error = normalizeModelGatewayError(cause, {
      provider: 'p', model: 'm', phase: 'stream_open',
    })
    expect(error.toJSON()).toMatchObject({ code: 'MODEL_CONNECTION_ERROR', retryable: true })
  })

  test('redacts credentials, URLs, prompt text, and raw provider bodies', () => {
    const cause = Object.assign(
      new Error('sk-secret failed at https://provider.example/v1 prompt=private'),
      { status: 500, headers: { authorization: 'Bearer secret' }, body: 'private response' },
    )
    const error = normalizeModelGatewayError(cause, {
      provider: 'volcengine', model: 'glm-5.2', phase: 'request',
    })

    const serialized = JSON.stringify(error)
    expect(serialized).not.toContain('sk-secret')
    expect(serialized).not.toContain('provider.example')
    expect(serialized).not.toContain('private')
    expect(serialized).not.toContain('authorization')
  })

  test('normalization is idempotent', () => {
    const original = new ModelGatewayError({
      code: 'MODEL_TIMEOUT', message: 'Model provider request timed out.',
      phase: 'request', provider: 'p', model: 'm', retryable: true,
    })
    expect(normalizeModelGatewayError(original, {
      provider: 'other', model: 'other', phase: 'response_parse',
    })).toBe(original)
  })
})
