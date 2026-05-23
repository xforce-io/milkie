// Stub database module — used only as a fixture for code-review tests
/* eslint-disable @typescript-eslint/no-explicit-any */

export async function raw(_query: string): Promise<any[]> {
  throw new Error('stub: not implemented')
}

export async function query(_sql: string, _params?: unknown[]): Promise<any[]> {
  throw new Error('stub: not implemented')
}
