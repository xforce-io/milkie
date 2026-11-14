/** Public native-search storage. Trace remains metadata-only. */
import { ExecutionError, type ExecutionRecord, type NativeWebSearchItem, type NativeWebSearchPage } from './types.js'
import type { ExecutionStore } from './store.js'
const MAX_BYTES = 2 * 1024 * 1024
const MAX_ITEMS = 64
function object(value: unknown): value is Record<string, unknown> { return !!value && typeof value === 'object' && !Array.isArray(value) }
function nullableText(value: unknown): boolean { return value === null || typeof value === 'string' }
function validAction(value: unknown): value is Record<string, unknown> | null {
  if (value === null) return true
  if (!object(value)) return false
  const fields = value.type === 'search' ? ['type', 'query', 'queries'] : value.type === 'openPage' ? ['type', 'url'] : value.type === 'findInPage' ? ['type', 'url', 'pattern'] : value.type === 'other' ? ['type'] : []
  if (fields.length === 0 || Object.keys(value).some(k => !fields.includes(k))) return false
  if (value.type === 'search') return nullableText(value.query) && (value.queries === null || Array.isArray(value.queries) && value.queries.every(q => typeof q === 'string'))
  if (value.type === 'openPage') return nullableText(value.url)
  if (value.type === 'findInPage') return nullableText(value.url) && nullableText(value.pattern)
  return true
}
function validateItems(items: unknown): items is NativeWebSearchItem[] {
  return Array.isArray(items) && items.length <= MAX_ITEMS && items.every(v => object(v) && typeof v.nativeItemId === 'string' && v.nativeItemId.length > 0 && v.nativeItemId.length <= 128 && ['started', 'completed'].includes(v.phase as string) && typeof v.query === 'string' && validAction(v.action) && (v.results === null || Array.isArray(v.results))) && new Set(items.map(v => v.nativeItemId)).size === items.length
}
export function initializeWebSearches(store: ExecutionStore, run: ExecutionRecord): void {
  if ((run.webSearch ?? 'disabled') !== 'disabled') store.write('native-searches', run.runId, { version: 1, runId: run.runId, contextId: run.contextId, mode: run.webSearch, items: [], complete: false })
}
export function readWebSearches(store: ExecutionStore, run: ExecutionRecord): NativeWebSearchPage {
  const mode = run.webSearch ?? 'disabled'
  if (!['disabled', 'cached', 'live'].includes(mode)) throw new ExecutionError('storage_error')
  const saved = store.read<NativeWebSearchPage>('native-searches', run.runId)
  if (!saved && mode !== 'disabled') throw new ExecutionError('storage_error')
  if (saved && (saved.runId !== run.runId || saved.contextId !== run.contextId || saved.mode !== mode || !validateItems(saved.items) || Buffer.byteLength(JSON.stringify(saved)) > MAX_BYTES)) throw new ExecutionError('storage_error')
  const items = saved?.items ?? []
  return { version: 1, runId: run.runId, contextId: run.contextId, mode, items, complete: run.status === 'succeeded' && run.finishedAt !== undefined && run.stopped && items.every(i => i.phase === 'completed') && run.code !== 'native_search_limit_exceeded' }
}
export function recordWebSearch(store: ExecutionStore, run: ExecutionRecord, item: Record<string, unknown>, phase: 'started' | 'completed'): void {
  if (typeof item.id !== 'string' || item.id.length < 1 || item.id.length > 128 || typeof item.query !== 'string' || !validAction(item.action) || !(item.results === null || Array.isArray(item.results))) throw new ExecutionError('protocol_error')
  const page = readWebSearches(store, run)
  const index = page.items.findIndex(i => i.nativeItemId === item.id)
  if (phase === 'started' ? index !== -1 : index === -1 || page.items[index]?.phase !== 'started') throw new ExecutionError('protocol_error')
  const value: NativeWebSearchItem = { nativeItemId: item.id, phase, query: item.query, action: item.action, results: item.results }
  if (phase === 'started') page.items.push(value)
  else page.items[index] = value
  page.complete = false
  if (page.items.length > MAX_ITEMS || Buffer.byteLength(JSON.stringify(page)) > MAX_BYTES) throw new ExecutionError('native_search_limit_exceeded')
  store.write('native-searches', run.runId, page)
}
