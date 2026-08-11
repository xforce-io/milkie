import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

type PackedFile = { path: string }
type PackResult = { filename: string; files: PackedFile[] }

const ROOT = path.resolve(__dirname, '../..')

function run(command: string, args: string[], cwd: string): string {
  return execFileSync(command, args, {
    cwd,
    encoding: 'utf8',
    env: { ...process.env, npm_config_audit: 'false', npm_config_fund: 'false' },
  })
}

describe('npm package distribution', () => {
  let tempRoot: string

  beforeEach(() => {
    tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'milkie-package-'))
  })

  afterEach(() => {
    fs.rmSync(tempRoot, { recursive: true, force: true })
    fs.rmSync(path.join(ROOT, 'dist'), { recursive: true, force: true })
  })

  it('builds a tarball that installs a runnable CLI consumer', () => {
    fs.rmSync(path.join(ROOT, 'dist'), { recursive: true, force: true })

    const packJson = run('npm', ['pack', '--json', '--pack-destination', tempRoot], ROOT)
    const packed = (JSON.parse(packJson) as PackResult[])[0]
    if (!packed) throw new Error('npm pack did not produce a tarball')
    const paths = packed.files.map(file => file.path)

    expect(paths).toEqual(expect.arrayContaining([
      'dist/index.js',
      'dist/cli/index.js',
      'agents/diagnoser.md',
    ]))

    const consumer = path.join(tempRoot, 'consumer')
    fs.mkdirSync(path.join(consumer, '.milkie'), { recursive: true })
    fs.mkdirSync(path.join(consumer, 'agents'), { recursive: true })
    fs.writeFileSync(path.join(consumer, 'package.json'), '{"private":true}\n')
    fs.writeFileSync(
      path.join(consumer, '.milkie', 'agents.json'),
      '{"agents":[{"id":"dataspace","file":"../agents/dataspace.md"}]}\n',
    )
    fs.writeFileSync(
      path.join(consumer, 'agents', 'dataspace.md'),
      '---\nagentId: dataspace\nfsm:\n  states: []\nmodel:\n  provider: stub\n  model: stub\n  adapter: stub\n---\nconsumer test\n',
    )

    run('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', path.join(tempRoot, packed.filename)], consumer)

    const binary = path.join(consumer, 'node_modules', '.bin', 'milkie')
    expect(run(binary, ['--help'], consumer)).toContain('Usage: milkie')
    expect(run(binary, ['agent', 'list'], consumer)).toContain('"id":"dataspace"')
  })
})
