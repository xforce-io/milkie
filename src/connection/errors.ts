import {
  SAFE_CONNECTION_MESSAGES,
  type ConnectionConfigCode,
} from './types.js'

export class ConnectionConfigError extends Error {
  readonly code: ConnectionConfigCode
  readonly fields: string[]

  constructor(code: ConnectionConfigCode, fields: string[]) {
    const unique = [...new Set(fields)].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))
    super(SAFE_CONNECTION_MESSAGES[code])
    this.name = 'ConnectionConfigError'
    this.code = code
    this.fields = unique
  }

  toJSON(): { code: ConnectionConfigCode; message: string; fields: string[] } {
    return { code: this.code, message: this.message, fields: this.fields }
  }
}

