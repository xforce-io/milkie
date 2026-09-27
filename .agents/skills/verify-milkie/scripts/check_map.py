#!/usr/bin/env python3
"""检查地图文件、相对链接和主要公开入口的归属；不验证产品行为。"""
from pathlib import Path
import re
import sys

base = Path(__file__).resolve().parents[1]
root = base.parents[2]
features = base / 'features'
index = (features / 'README.md').read_text()
errors = []

for file in base.rglob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)', file.read_text()):
        if '://' in target or target.startswith('#'):
            continue
        path = target.split('#', 1)[0]
        if not (file.parent / path).exists():
            errors.append(f'{file.relative_to(root)}: 无效文件引用 {target}')

cards = sorted(p for p in features.glob('*.md') if p.name != 'README.md')
for file in cards:
    if f']({file.name})' not in index:
        errors.append(f'功能文件不在索引: {file.name}')

ledger = {'SDK': {}, 'CLI': {}, 'HTTP': {}}
for kind, entry, target in re.findall(r'^\| (SDK|CLI|HTTP) \| `([^`]+)` \| \[[^\]]+\]\(([^)]+)\) \|$', index, re.M):
    if entry in ledger[kind]:
        errors.append(f'重复入口: {kind} {entry}')
    ledger[kind][entry] = target
    if not (features / target).is_file():
        errors.append(f'入口指向不存在的功能: {entry}: {target}')

milkie = (root / 'src/runtime/Milkie.ts').read_text().split('export class Milkie', 1)[1]
sdk = set(re.findall(r'^  (?:async )?(\w+)\(', milkie, re.M)) - {'constructor'}
cli_source = (root / 'src/cli/main.ts').read_text()
cli = set()
# This source declares the agent group, then trace group, then standalone serve.
for section, prefix in [(cli_source.split("const agent = program.command('agent')", 1)[1].split("const trace = program.command('trace')", 1)[0], 'agent '),
                        (cli_source.split("const trace = program.command('trace')", 1)[1].split("  program\n    .command('serve')", 1)[0], 'trace ')]:
    cli.update(prefix + name for name in re.findall(r"\.command\('([^']+)'\)", section))
if ".command('serve')" in cli_source:
    cli.add('serve')
http_source = (root / 'src/cli/serve.ts').read_text()
http = {f'{method} {route}' for method, route in re.findall(r"req.method === '(GET|POST|PUT|PATCH|DELETE)'\s*&& route === '([^']+)'", http_source)}

for kind, actual in [('SDK', sdk), ('CLI', cli), ('HTTP', http)]:
    documented = set(ledger[kind])
    for item in sorted(actual - documented):
        errors.append(f'源码入口未登记: {kind} {item}')
    for item in sorted(documented - actual):
        errors.append(f'登记入口未在解析范围找到: {kind} {item}')
    print(f'{kind}: {len(actual)} 个源码入口，{len(documented)} 个已登记入口')

# Commands in the cards must point to real suites, not old Story-reserved names.
for file in cards:
    for test in set(re.findall(r'(?<![\w/])((?:src/__tests__|tests/e2e)/[\w./-]+\.test\.ts)', file.read_text())):
        if not (root / test).is_file():
            errors.append(f'{file.name}: 测试不存在 {test}')

# A Story row without a capability assignment is not a completed migration map.
for story in sorted((root / 'docs/stories').glob('s-*.md')):
    rows = [line for line in index.splitlines() if f'/docs/stories/{story.name})' in line]
    if len(rows) != 1 or not re.search(r'\]\(\d\d-[^)]+\.md\)', rows[0]):
        errors.append(f'Story 未唯一映射到功能文件: {story.name}')

print(f'功能文件: {len(cards)}；错误: {len(errors)}')
for error in errors:
    print(error, file=sys.stderr)
sys.exit(bool(errors))
