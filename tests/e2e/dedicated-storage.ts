import { chmodSync, copyFileSync, existsSync, mkdirSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

/** Copy host-prepared login material into a fresh config directory. Does not read or log the file. */
export function prepareDedicatedStorage(root: string, runtime: string): { configDir: string; sessionDir: string } {
  const configDir = join(root, 'config')
  const sessionDir = join(root, 'sessions')
  mkdirSync(configDir, { recursive: true, mode: 0o700 })
  mkdirSync(sessionDir, { recursive: true, mode: 0o700 })
  const source = runtime === 'grok-cli' ? join(homedir(), '.grok', 'auth.json') : join(homedir(), '.pi', 'agent', 'auth.json')
  if (!existsSync(source)) throw new Error(`Login material is not prepared for ${runtime}.`)
  const target = join(configDir, 'auth.json')
  copyFileSync(source, target)
  chmodSync(target, 0o600)
  if (runtime === 'pi') {
    const settings = join(homedir(), '.pi', 'agent', 'settings.json')
    if (existsSync(settings)) copyFileSync(settings, join(configDir, 'settings.json'))
  }
  return { configDir, sessionDir }
}
