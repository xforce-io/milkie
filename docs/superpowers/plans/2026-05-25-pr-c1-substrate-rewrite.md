# PR-C1: AgentRuntime substrate rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing `ContextLayer` with `ContextRegions + assemble` (landed in PR-A and PR-B) inside `AgentRuntime`. Split scratchpad from history. Wire intra-turn and inter-turn (crystallization) boundary engines. Re-record the two versionable replay fixtures. Delete `ContextLayer.ts`.

**Architecture:** AgentRuntime holds a `ContextRegions` store instead of a `ContextLayer`. On every LLM call, `assemble(regions, scope)` produces the `system/messages/tools` triple; AgentRuntime composes that with `model` from agent config to make the `ModelRequest`. Every assistant response and tool result becomes a scratchpad region (`interTurn='turn-local'`). On turn end (`run`/`continueTurn` returning), `runInterTurnEngine` crystallizes: extracts the final assistant text + current-turn user input into a single history-pair region, drops all turn-local regions.

**Tech Stack:** TypeScript, jest (`ts-jest`), existing `RecordingIOPort`/`ReplayingIOPort` for byte-identical replay.

**Spec:** `docs/superpowers/specs/2026-05-25-context-region-substrate-design.md` §4.2–§4.4, §5, §7, §11.

---

## File Structure

**Create:**
- `src/context/lifecycleEngine.ts` — `runIntraTurnEngine`, `runInterTurnEngine`, helpers (`extractFinalAssistantText`, `makeScratchpadRegion`, `makeHistoryPairRegion`, `makeCurrentTurnRegion`, `makeHeaderRegion`, `makeSkillRegion`, `makeStateInstructionsRegion`, `makeWmRegion`, `makeToolSchemaRegion`)
- `src/__tests__/lifecycleEngine.test.ts` — unit tests for both engines + helpers

**Modify:**
- `src/runtime/AgentRuntime.ts` — replace `ContextLayer` field + 14 call sites
- `src/__tests__/Replay.nondet.test.ts` — uses `ContextLayer` constructor directly; rewrite to use new substrate
- `src/index.ts` — remove `ContextLayer` re-export
- `examples/s-005-replay/.milkie/runs/8da9174a-567d-406a-9596-9ee53947b194.jsonl` — re-record
- `examples/s-002-inspect/.milkie/runs/1e65a3ec-03d4-40b4-9546-983fce9cb0e3.jsonl` — re-record

**Delete:**
- `src/context/ContextLayer.ts`

---

## Task 1: Lifecycle engine — `extractFinalAssistantText` helper

**Files:**
- Create: `src/context/lifecycleEngine.ts`
- Test: `src/__tests__/lifecycleEngine.test.ts`

**Why:** Crystallization needs to pull out the last assistant text from scratchpad regions. Isolating this helper makes it testable in isolation and reusable.

- [ ] **Step 1: Write the failing test**

Create `src/__tests__/lifecycleEngine.test.ts`:

```typescript
import { ContextRegions } from '../context/ContextRegions'
import { extractFinalAssistantText } from '../context/lifecycleEngine'
import type { RegionInput } from '../context/Region'

function scratchAssistantRegion(text: string, ordinal: number, hasToolUse = false): RegionInput {
  return {
    target:    'message',
    section:   'scratchpad',
    intraTurn: 'turn-persistent',
    interTurn: 'turn-local',
    stability: 'volatile',
    ordinal,
    content:   { role: 'assistant', text, hasToolUse },
    format:    () => ({ role: 'assistant', content: [] }),
  }
}

describe('extractFinalAssistantText', () => {
  test('returns empty string when no scratchpad regions', () => {
    const r = new ContextRegions(() => 0)
    expect(extractFinalAssistantText(r)).toBe('')
  })

  test('returns the latest assistant region without tool_use', () => {
    const r = new ContextRegions(() => 0)
    r.set('s1', scratchAssistantRegion('first answer', 1, false))
    r.set('s2', scratchAssistantRegion('second answer', 2, false))
    expect(extractFinalAssistantText(r)).toBe('second answer')
  })

  test('skips assistant regions that contain tool_use', () => {
    const r = new ContextRegions(() => 0)
    r.set('s1', scratchAssistantRegion('thinking...', 1, true))
    r.set('s2', scratchAssistantRegion('final answer', 2, false))
    expect(extractFinalAssistantText(r)).toBe('final answer')
  })

  test('ignores non-scratchpad regions', () => {
    const r = new ContextRegions(() => 0)
    r.set('hist', {
      target: 'message', section: 'history',
      intraTurn: 'turn-persistent', interTurn: 'session-persistent', stability: 'session-stable',
      content: { role: 'assistant', text: 'old', hasToolUse: false },
      format: () => ({ role: 'assistant', content: [] }),
    })
    expect(extractFinalAssistantText(r)).toBe('')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest src/__tests__/lifecycleEngine.test.ts`
Expected: FAIL with `Cannot find module '../context/lifecycleEngine'`

- [ ] **Step 3: Create lifecycleEngine.ts with extractFinalAssistantText**

Create `src/context/lifecycleEngine.ts`:

```typescript
// Boundary engines for the context region substrate.
// Spec: docs/superpowers/specs/2026-05-25-context-region-substrate-design.md §7
//
// Two engines fire at distinct boundaries:
//   - runIntraTurnEngine: at each FSM step boundary (applies pending mutations,
//     handles state-scoped expiry, tool-buffer/one-shot decrement, TTL expiry)
//   - runInterTurnEngine: at turn-end (crystallization — archive final answer
//     into history pair region; drop turn-local; promote-to-wm)

import type { ContextRegions } from './ContextRegions'

interface ScratchAssistantContent {
  role:       'assistant'
  text:       string
  hasToolUse: boolean
}

export function extractFinalAssistantText(regions: ContextRegions): string {
  const candidates = [...regions._allRegions()]
    .filter(r => r.section === 'scratchpad')
    .filter(r => (r.content as { role?: string }).role === 'assistant')
    .sort((a, b) => b.createdAt - a.createdAt)
  for (const r of candidates) {
    const c = r.content as ScratchAssistantContent
    if (!c.hasToolUse && c.text) return c.text
  }
  return ''
}
```

- [ ] **Step 4: Run tests to verify GREEN**

Run: `npx jest src/__tests__/lifecycleEngine.test.ts`
Expected: 4 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/context/lifecycleEngine.ts src/__tests__/lifecycleEngine.test.ts
git commit -m "feat(context): extractFinalAssistantText helper (PR-C1 step 1/13)"
```

---

## Task 2: Lifecycle engine — region factory helpers

**Files:**
- Modify: `src/context/lifecycleEngine.ts`
- Test: `src/__tests__/lifecycleEngine.test.ts`

**Why:** AgentRuntime will need to construct many region inputs at specific points (insert assistant response, insert tool result, insert skill, etc). Centralizing these factories in lifecycleEngine keeps AgentRuntime focused on orchestration.

- [ ] **Step 1: Write the failing test (append to lifecycleEngine.test.ts)**

```typescript
import {
  extractFinalAssistantText,
  makeScratchpadAssistantRegion,
  makeScratchpadToolResultRegion,
  makeCurrentTurnRegion,
  makeHistoryPairRegion,
  makeHeaderRegion,
  makeSkillRegion,
  makeStateInstructionsRegion,
  makeWmRegion,
  makeToolSchemaRegion,
} from '../context/lifecycleEngine'
import type { MessageContent } from '../types/common.js'
import type { ToolSchema } from '../types/model.js'

describe('region factories', () => {
  test('makeHeaderRegion: target=system, section=header, immutable, session-persistent', () => {
    const r = makeHeaderRegion('You are an agent.')
    expect(r.target).toBe('system')
    expect(r.section).toBe('header')
    expect(r.stability).toBe('immutable')
    expect(r.interTurn).toBe('session-persistent')
    expect(r.format(r.content)).toBe('You are an agent.')
  })

  test('makeSkillRegion(turn): target=system, section=session-skills, interTurn=turn-local', () => {
    const r = makeSkillRegion('verifier', 'INSTRUCTIONS', 'turn')
    expect(r.section).toBe('session-skills')
    expect(r.interTurn).toBe('turn-local')
    expect(r.format(r.content)).toContain('verifier')
    expect(r.format(r.content)).toContain('INSTRUCTIONS')
  })

  test('makeSkillRegion(session): section=persistent-skills, interTurn=session-persistent', () => {
    const r = makeSkillRegion('helper', 'INST', 'session')
    expect(r.section).toBe('persistent-skills')
    expect(r.interTurn).toBe('session-persistent')
  })

  test('makeStateInstructionsRegion: state-scoped intraTurn, section=state', () => {
    const r = makeStateInstructionsRegion('researching', 'Focus on the chapter content.')
    expect(r.section).toBe('state')
    expect(r.intraTurn).toEqual({ kind: 'state-scoped', state: 'researching' })
    expect(r.format(r.content)).toContain('Focus on the chapter content.')
  })

  test('makeWmRegion: section=wm, deterministic key order in JSON', () => {
    const r1 = makeWmRegion({ b: 2, a: 1, c: 3 }, [])
    const r2 = makeWmRegion({ a: 1, b: 2, c: 3 }, [])
    // Same data, different insertion order → same serialized output (sorted keys)
    expect(r1.format(r1.content)).toBe(r2.format(r2.content))
  })

  test('makeWmRegion: omitted when data + log are both empty', () => {
    const r = makeWmRegion({}, [])
    expect(r).toBeNull()
  })

  test('makeCurrentTurnRegion: target=message, section=current-turn, turn-local', () => {
    const r = makeCurrentTurnRegion('hello')
    expect(r.target).toBe('message')
    expect(r.section).toBe('current-turn')
    expect(r.interTurn).toBe('turn-local')
    const msg = r.format(r.content) as { role: string; content: MessageContent[] }
    expect(msg.role).toBe('user')
    expect((msg.content[0] as { text: string }).text).toBe('hello')
  })

  test('makeScratchpadAssistantRegion: turn-local, section=scratchpad, role=assistant', () => {
    const content: MessageContent[] = [{ type: 'text', text: 'thinking' }]
    const r = makeScratchpadAssistantRegion(content, false)
    expect(r.section).toBe('scratchpad')
    expect(r.interTurn).toBe('turn-local')
    const msg = r.format(r.content) as { role: string; content: MessageContent[] }
    expect(msg.role).toBe('assistant')
    expect(msg.content).toEqual(content)
  })

  test('makeScratchpadToolResultRegion: tool message with tool_result content', () => {
    const content: MessageContent[] = [
      { type: 'tool_result', tool_use_id: 'tc1', content: 'ok' },
    ]
    const r = makeScratchpadToolResultRegion(content)
    expect(r.section).toBe('scratchpad')
    expect(r.interTurn).toBe('turn-local')
    const msg = r.format(r.content) as { role: string; content: MessageContent[] }
    expect(msg.role).toBe('tool')
  })

  test('makeHistoryPairRegion: returns Message[] (user + assistant)', () => {
    const r = makeHistoryPairRegion('what time?', 'noon')
    expect(r.section).toBe('history')
    expect(r.interTurn).toBe('session-persistent')
    const msgs = r.format(r.content) as { role: string; content: MessageContent[] }[]
    expect(msgs).toHaveLength(2)
    expect(msgs[0]!.role).toBe('user')
    expect((msgs[0]!.content[0] as { text: string }).text).toBe('what time?')
    expect(msgs[1]!.role).toBe('assistant')
    expect((msgs[1]!.content[0] as { text: string }).text).toBe('noon')
  })

  test('makeToolSchemaRegion: target=tool, section=default', () => {
    const schema: ToolSchema = { name: 'echo', description: 'e', inputSchema: {} }
    const r = makeToolSchemaRegion(schema)
    expect(r.target).toBe('tool')
    expect(r.section).toBe('default')
    expect(r.format(r.content)).toBe(schema)
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest src/__tests__/lifecycleEngine.test.ts`
Expected: FAIL — factories missing

- [ ] **Step 3: Add factories to lifecycleEngine.ts**

Append to `src/context/lifecycleEngine.ts`:

```typescript
import type { Region, RegionInput } from './Region'
import type { Message, MessageContent } from '../types/common.js'
import type { ToolSchema } from '../types/model.js'

export function makeHeaderRegion(systemPrompt: string): RegionInput {
  return {
    target:    'system',
    section:   'header',
    intraTurn: 'turn-persistent',
    interTurn: 'session-persistent',
    stability: 'immutable',
    content:   systemPrompt,
    format:    (c) => String(c),
  }
}

export function makeSkillRegion(
  name: string,
  instructions: string,
  scope: 'turn' | 'session' = 'session',
): RegionInput {
  return {
    target:    'system',
    section:   scope === 'session' ? 'persistent-skills' : 'session-skills',
    intraTurn: 'turn-persistent',
    interTurn: scope === 'session' ? 'session-persistent' : 'turn-local',
    stability: scope === 'session' ? 'session-stable' : 'turn-stable',
    content:   { name, instructions },
    format:    (c) => {
      const { name, instructions } = c as { name: string; instructions: string }
      return `\n--- Skill: ${name} ---\n${instructions}`
    },
  }
}

export function makeStateInstructionsRegion(
  state: string,
  instructions: string,
): RegionInput {
  return {
    target:    'system',
    section:   'state',
    intraTurn: { kind: 'state-scoped', state },
    interTurn: 'turn-local',
    stability: 'turn-stable',
    content:   instructions,
    format:    (c) => `\n--- Current Instructions ---\n${String(c)}`,
  }
}

export function makeWmRegion(
  data: Record<string, unknown>,
  log: unknown[],
): RegionInput | null {
  if (Object.keys(data).length === 0 && log.length === 0) return null
  // Sort keys for deterministic byte-identical output across runs.
  const sortedData: Record<string, unknown> = {}
  for (const k of Object.keys(data).sort()) sortedData[k] = data[k]
  return {
    target:    'system',
    section:   'wm',
    intraTurn: 'turn-persistent',
    interTurn: 'session-persistent',
    stability: 'volatile',
    content:   { data: sortedData, log },
    format:    (c) =>
      '\n--- Working Memory ---\n' + JSON.stringify(c, null, 2),
  }
}

export function makeCurrentTurnRegion(input: string): RegionInput {
  return {
    target:    'message',
    section:   'current-turn',
    intraTurn: 'turn-persistent',
    interTurn: 'turn-local',
    stability: 'volatile',
    content:   input,
    format:    (c): Message => ({
      role:    'user',
      content: [{ type: 'text', text: String(c) }],
    }),
  }
}

export function makeScratchpadAssistantRegion(
  content: MessageContent[],
  hasToolUse: boolean,
): RegionInput {
  // hasToolUse stored on the region's content metadata so extractFinalAssistantText
  // can skip intermediate (tool-bearing) assistant turns without parsing the message.
  // The text field is the concatenation of any text parts (used by crystallization).
  const text = content
    .filter(c => c.type === 'text')
    .map(c => (c as { text: string }).text)
    .join('')
  return {
    target:    'message',
    section:   'scratchpad',
    intraTurn: 'turn-persistent',
    interTurn: 'turn-local',
    stability: 'volatile',
    content:   { role: 'assistant' as const, text, hasToolUse, raw: content },
    format:    (c): Message => ({
      role:    'assistant',
      content: (c as { raw: MessageContent[] }).raw,
    }),
  }
}

export function makeScratchpadToolResultRegion(
  content: MessageContent[],
): RegionInput {
  return {
    target:    'message',
    section:   'scratchpad',
    intraTurn: 'turn-persistent',
    interTurn: 'turn-local',
    stability: 'volatile',
    content:   { role: 'tool' as const, raw: content },
    format:    (c): Message => ({
      role:    'tool',
      content: (c as { raw: MessageContent[] }).raw,
    }),
  }
}

export function makeHistoryPairRegion(
  userInput: string,
  assistantText: string,
): RegionInput {
  return {
    target:    'message',
    section:   'history',
    intraTurn: 'turn-persistent',
    interTurn: 'session-persistent',
    stability: 'session-stable',
    content:   { userInput, assistantText },
    format:    (c): Message[] => {
      const { userInput, assistantText } = c as { userInput: string; assistantText: string }
      return [
        { role: 'user',      content: [{ type: 'text', text: userInput }] },
        { role: 'assistant', content: [{ type: 'text', text: assistantText }] },
      ]
    },
  }
}

export function makeToolSchemaRegion(schema: ToolSchema): RegionInput {
  return {
    target:    'tool',
    section:   'default',
    intraTurn: 'turn-persistent',
    interTurn: 'session-persistent',
    stability: 'session-stable',
    content:   schema,
    format:    (c) => c as ToolSchema,
  }
}
```

- [ ] **Step 4: Run tests to verify GREEN**

Run: `npx jest src/__tests__/lifecycleEngine.test.ts`
Expected: 14 tests pass (4 from Task 1 + 10 factories)

- [ ] **Step 5: Commit**

```bash
git add src/context/lifecycleEngine.ts src/__tests__/lifecycleEngine.test.ts
git commit -m "feat(context): region factory helpers (PR-C1 step 2/13)"
```

---

## Task 3: Lifecycle engine — `runInterTurnEngine` crystallization

**Files:**
- Modify: `src/context/lifecycleEngine.ts`
- Test: `src/__tests__/lifecycleEngine.test.ts`

**Why:** Turn-end crystallization is the heart of the substrate. Tested in isolation here; AgentRuntime calls it later (Task 9).

- [ ] **Step 1: Append failing tests**

```typescript
import { runInterTurnEngine } from '../context/lifecycleEngine'

describe('runInterTurnEngine — turn-end crystallization', () => {
  test('archives (user, finalAssistant) pair into a history region', () => {
    const r = new ContextRegions(() => 100)
    r.set('current', makeCurrentTurnRegion('what time?'))
    r.set('s-final', makeScratchpadAssistantRegion(
      [{ type: 'text', text: 'noon' }],
      false,
    ))
    runInterTurnEngine(r, { boundary: 'turn-end', userInput: 'what time?', now: 999 })
    const histIds = [...r._allRegions()]
      .filter(x => x.section === 'history')
      .map(x => x.id)
    expect(histIds).toHaveLength(1)
    const hist = r.get(histIds[0]!)!
    const msgs = hist.format(hist.content) as Message[]
    expect((msgs[0]!.content[0] as { text: string }).text).toBe('what time?')
    expect((msgs[1]!.content[0] as { text: string }).text).toBe('noon')
  })

  test('drops all turn-local regions (scratchpad + current-turn + turn-scope skills)', () => {
    const r = new ContextRegions(() => 0)
    r.set('current', makeCurrentTurnRegion('q'))
    r.set('s1', makeScratchpadAssistantRegion([{ type: 'text', text: 'a' }], false))
    r.set('skill-turn', makeSkillRegion('v', 'I', 'turn'))
    runInterTurnEngine(r, { boundary: 'turn-end', userInput: 'q', now: 1 })
    expect(r.get('current')).toBeUndefined()
    expect(r.get('s1')).toBeUndefined()
    expect(r.get('skill-turn')).toBeUndefined()
  })

  test('keeps session-persistent regions (header, session skills, history)', () => {
    const r = new ContextRegions(() => 0)
    r.set('hdr', makeHeaderRegion('agent'))
    r.set('skill-s', makeSkillRegion('s', 'I', 'session'))
    r.set('current', makeCurrentTurnRegion('q'))
    r.set('s1', makeScratchpadAssistantRegion([{ type: 'text', text: 'a' }], false))
    runInterTurnEngine(r, { boundary: 'turn-end', userInput: 'q', now: 1 })
    expect(r.get('hdr')).toBeDefined()
    expect(r.get('skill-s')).toBeDefined()
  })

  test('TTL region dropped when now > deadline', () => {
    const r = new ContextRegions(() => 0)
    r.set('ttl', {
      target: 'system', section: 'wm',
      intraTurn: 'turn-persistent',
      interTurn: { kind: 'ttl', deadline: 100 },
      stability: 'volatile',
      content: 'expires',
      format: (c) => String(c),
    })
    runInterTurnEngine(r, { boundary: 'turn-end', userInput: 'q', now: 200 })
    expect(r.get('ttl')).toBeUndefined()
  })

  test('promote-to-wm region transforms into wm region (target=system, section=wm)', () => {
    const r = new ContextRegions(() => 0)
    r.set('learn', {
      target: 'message', section: 'scratchpad',
      intraTurn: 'turn-persistent',
      interTurn: 'promote-to-wm',
      stability: 'volatile',
      content: 'learned-fact',
      format: (c) => ({ role: 'tool', content: [{ type: 'text', text: String(c) }] }),
    })
    runInterTurnEngine(r, { boundary: 'turn-end', userInput: 'q', now: 1 })
    expect(r.get('learn')).toBeUndefined()
    const promoted = [...r._allRegions()].filter(x => x.id.startsWith('wm:'))
    expect(promoted).toHaveLength(1)
    expect(promoted[0]!.target).toBe('system')
    expect(promoted[0]!.section).toBe('wm')
    expect(promoted[0]!.interTurn).toBe('session-persistent')
  })

  test('userInput is optional — no archive when missing (e.g. interrupt save)', () => {
    const r = new ContextRegions(() => 0)
    r.set('s1', makeScratchpadAssistantRegion([{ type: 'text', text: 'a' }], false))
    runInterTurnEngine(r, { boundary: 'turn-end', now: 1 })
    const histIds = [...r._allRegions()].filter(x => x.section === 'history').map(x => x.id)
    expect(histIds).toEqual([])
    // Scratchpad still cleared because it's turn-local.
    expect(r.get('s1')).toBeUndefined()
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest src/__tests__/lifecycleEngine.test.ts -t runInterTurnEngine`
Expected: FAIL — `runInterTurnEngine` is not a function

- [ ] **Step 3: Implement runInterTurnEngine**

Append to `src/context/lifecycleEngine.ts`:

```typescript
export interface InterTurnContext {
  boundary:   'turn-end' | 'turn-start'
  userInput?: string
  now:        number
}

export interface CrystallizationSummary {
  kept:         string[]
  dropped:      string[]
  promoted:     Array<{ from: string; to: string }>
  archivedPair: string | undefined
}

export function runInterTurnEngine(
  regions: ContextRegions,
  ctx: InterTurnContext,
): { crystallization?: CrystallizationSummary } {
  if (ctx.boundary !== 'turn-end') return {}

  const summary: CrystallizationSummary = {
    kept: [], dropped: [], promoted: [], archivedPair: undefined,
  }

  // Step 1: archive final answer (if userInput available)
  if (ctx.userInput !== undefined) {
    const finalText = extractFinalAssistantText(regions)
    const pairId = `history:turn-${ctx.now}`
    regions.set(pairId, makeHistoryPairRegion(ctx.userInput, finalText))
    summary.archivedPair = pairId
  }

  // Step 2: iterate snapshot of all regions, apply per-interTurn rule.
  // Use a snapshot (`[...regions._allRegions()]`) because we mutate during iteration.
  for (const r of [...regions._allRegions()]) {
    if (r.id === summary.archivedPair) {
      summary.kept.push(r.id)
      continue
    }
    if (r.interTurn === 'session-persistent') {
      summary.kept.push(r.id)
      continue
    }
    if (r.interTurn === 'turn-local') {
      regions.delete(r.id)
      summary.dropped.push(r.id)
      continue
    }
    if (typeof r.interTurn === 'object' && r.interTurn.kind === 'ttl') {
      if (ctx.now > r.interTurn.deadline) {
        regions.delete(r.id)
        summary.dropped.push(r.id)
      } else {
        summary.kept.push(r.id)
      }
      continue
    }
    if (r.interTurn === 'promote-to-wm') {
      const promotedId = `wm:${r.id}`
      regions.set(promotedId, {
        target:    'system',
        section:   'wm',
        intraTurn: 'turn-persistent',
        interTurn: 'session-persistent',
        stability: 'session-stable',
        content:   r.content,
        format:    r.format,
      })
      regions.delete(r.id)
      summary.promoted.push({ from: r.id, to: promotedId })
      continue
    }
    if (r.interTurn === 'summarize-on-overflow') {
      // Phase 1: treat as session-persistent; budget summarization is future work.
      summary.kept.push(r.id)
      continue
    }
  }

  return { crystallization: summary }
}
```

- [ ] **Step 4: Run all lifecycleEngine tests**

Run: `npx jest src/__tests__/lifecycleEngine.test.ts`
Expected: 20 tests pass (4 + 10 + 6)

- [ ] **Step 5: Commit**

```bash
git add src/context/lifecycleEngine.ts src/__tests__/lifecycleEngine.test.ts
git commit -m "feat(context): runInterTurnEngine crystallization (PR-C1 step 3/13)"
```

---

## Task 4: AgentRuntime — swap ContextLayer field to ContextRegions (header + WM only)

**Files:**
- Modify: `src/runtime/AgentRuntime.ts`

**Why:** Switch internal storage WITHOUT changing AgentRuntime's external behavior — keep buildRequest/appendHistory delegating to a thin shim until later tasks. Lets us verify each step in isolation.

**Approach:** This task is unusual — we do NOT replace any call sites yet. We just stand up `this.regions: ContextRegions` alongside `this.context: ContextLayer`, populate header region, and verify existing tests still pass.

- [ ] **Step 1: Add ContextRegions field + helpers in constructor**

Modify `src/runtime/AgentRuntime.ts` imports (line 9 area):

```typescript
import { ContextLayer } from '../context/ContextLayer.js'   // keep — removed in Task 11
import { ContextRegions } from '../context/ContextRegions.js'
import { makeHeaderRegion } from '../context/lifecycleEngine.js'
```

Add field declaration after `private readonly context: ContextLayer` (around line 48):

```typescript
private readonly regions: ContextRegions
```

In constructor after `this.context = new ContextLayer({...})` (around line 80) add:

```typescript
this.regions = new ContextRegions(() => this.ioPort.now())
this.regions.set('header', makeHeaderRegion(opts.config.systemPrompt))
```

- [ ] **Step 2: Verify nothing breaks**

Run: `npm run test:unit`
Expected: 30 tests pass (no behavior changed — `regions` exists but isn't consumed yet)

Run: `npm run test:e2e:deterministic`
Expected: 7 pass + 15 skipped

- [ ] **Step 3: Commit**

```bash
git add src/runtime/AgentRuntime.ts
git commit -m "feat(runtime): add ContextRegions field alongside ContextLayer (PR-C1 step 4/13)"
```

---

## Task 5: AgentRuntime — replace `setCurrentTurn` with current-turn region

**Files:**
- Modify: `src/runtime/AgentRuntime.ts`

**Why:** First call-site swap. setCurrentTurn is called in `run` (line 325) and `continueTurn` (line 363); read in `runActionState` (line 509).

- [ ] **Step 1: Add import + helper**

Add to imports:

```typescript
import { makeCurrentTurnRegion } from '../context/lifecycleEngine.js'
```

Add private helper method after `applyPendingSkills`:

```typescript
private setCurrentTurn(input: string): void {
  this.regions.set('current-turn', makeCurrentTurnRegion(input))
  // Keep ContextLayer in sync until Task 6 removes its history role.
  this.context.setCurrentTurn(input)
}

private getCurrentTurn(): string | null {
  const r = this.regions.get('current-turn')
  if (!r) return null
  return r.content as string
}
```

- [ ] **Step 2: Replace direct ContextLayer calls**

Line 325: `this.context.setCurrentTurn(\`Goal: ${this.goal}\n\n${input}\`)` →
`this.setCurrentTurn(\`Goal: ${this.goal}\n\n${input}\`)`

Line 363: `this.context.setCurrentTurn(input)` →
`this.setCurrentTurn(input)`

Line 509: `this.context.currentTurn ?? ''` →
`this.getCurrentTurn() ?? ''`

Line 279 (saveCheckpoint): `this.context.currentTurn` →
`this.getCurrentTurn()`

- [ ] **Step 3: Verify**

Run: `npm run test:unit && npm run test:e2e:deterministic`
Expected: 30 + 7 pass

- [ ] **Step 4: Commit**

```bash
git add src/runtime/AgentRuntime.ts
git commit -m "feat(runtime): route current-turn through ContextRegions (PR-C1 step 5/13)"
```

---

## Task 6: AgentRuntime — replace `appendHistory` with scratchpad-region inserts

**Files:**
- Modify: `src/runtime/AgentRuntime.ts`

**Why:** This is the conceptual split. Old: every assistant + tool result goes into a single linear `history`. New: they all become individual scratchpad regions (turn-local). The cross-turn history is built later by crystallization.

- [ ] **Step 1: Add imports + helpers**

Add to imports:

```typescript
import {
  makeScratchpadAssistantRegion,
  makeScratchpadToolResultRegion,
} from '../context/lifecycleEngine.js'
```

Add private helpers:

```typescript
private appendScratchpadAssistant(content: MessageContent[]): void {
  const id = `scratch:${this.ioPort.uuid()}`
  const hasToolUse = content.some(c => c.type === 'tool_use')
  this.regions.set(id, makeScratchpadAssistantRegion(content, hasToolUse))
  // Keep ContextLayer in sync until Task 7 removes its buildRequest role.
  this.context.appendHistory({ role: 'assistant', content })
}

private appendScratchpadToolResults(content: MessageContent[]): void {
  const id = `scratch:${this.ioPort.uuid()}`
  this.regions.set(id, makeScratchpadToolResultRegion(content))
  this.context.appendHistory({ role: 'tool', content })
}
```

- [ ] **Step 2: Replace call sites**

Line 430: `this.context.appendHistory({ role: 'assistant', content: response.content })` →
`this.appendScratchpadAssistant(response.content)`

Line 454: `this.context.appendHistory({ role: 'tool', content: toolResultContent })` →
`this.appendScratchpadToolResults(toolResultContent)`

- [ ] **Step 3: Verify**

Run: `npm run test:unit && npm run test:e2e:deterministic`
Expected: 30 + 7 pass (still no behavior change — duplicate writes to both stores)

- [ ] **Step 4: Commit**

```bash
git add src/runtime/AgentRuntime.ts
git commit -m "feat(runtime): route scratchpad through ContextRegions (PR-C1 step 6/13)"
```

---

## Task 7: AgentRuntime — replace `buildRequest` with `assemble`

**Files:**
- Modify: `src/runtime/AgentRuntime.ts`

**Why:** Switch the actual LLM request construction to use the new substrate. THIS task changes byte-identical output of LLM requests; replay fixtures break here.

- [ ] **Step 1: Add imports + helpers**

Add to imports:

```typescript
import { assemble, type AssembleScope } from '../context/assemble.js'
import {
  makeStateInstructionsRegion,
  makeWmRegion,
  makeToolSchemaRegion,
  makeSkillRegion,
} from '../context/lifecycleEngine.js'
```

Add private helper to refresh all "volatile" regions (state-scope instructions, wm, tools) before each LLM call:

```typescript
private refreshTransientRegions(state: FSMState, schemas: ToolSchema[]): void {
  // State instructions — re-set every step so changes in state propagate.
  // The state-scoped intraTurn filter means only the current state's region
  // is visible during assembly, but we still need to put it there.
  if (state.instructions) {
    this.regions.set(
      `state-instr:${state.name}`,
      makeStateInstructionsRegion(state.name, state.instructions),
    )
  }

  // Working memory snapshot — written every step (deterministic key order in factory).
  const wmJson = this.memory.toJSON() as { data: Record<string, unknown>; log: unknown[] }
  const wmRegion = makeWmRegion(wmJson.data, wmJson.log)
  if (wmRegion) this.regions.set('wm', wmRegion)
  else this.regions.delete('wm')

  // Tool schemas — re-set per step in case state.tools restricted the set.
  // Clear existing tool regions first (only current schema set is valid).
  for (const r of [...this.regions._allRegions()].filter(r => r.target === 'tool')) {
    this.regions.delete(r.id)
  }
  for (const s of schemas) {
    this.regions.set(`tool:${s.name}`, makeToolSchemaRegion(s))
  }
}
```

Add import:

```typescript
import type { ToolSchema } from '../types/model.js'
```

- [ ] **Step 2: Replace buildRequest call (line 410 area)**

Replace these lines:

```typescript
const tools    = this.registry.getForState(state.tools)
const schemas  = this.registry.toSchemas(tools)
const request  = this.context.buildRequest(schemas, this.memory, state.instructions)
```

with:

```typescript
const tools   = this.registry.getForState(state.tools)
const schemas = this.registry.toSchemas(tools)

this.refreshTransientRegions(state, schemas)

const scope: AssembleScope = {
  currentState:  state.name,
  currentTurnId: `turn-${this.turnNumber}`,
  currentEpoch:  this.regions.getEpoch(),
}
const assembled = assemble(this.regions, scope)
const request   = {
  model:    this.config.model.model,
  system:   assembled.system,
  messages: assembled.messages,
  ...(assembled.tools ? { tools: assembled.tools } : {}),
}
```

- [ ] **Step 3: Update span attributes (line 417-418)**

Replace:

```typescript
loadedSkills: this.context.getLoadedInstructions(),
contextEpoch: this.context.getContextEpoch(),
```

with:

```typescript
loadedSkills: [...this.regions._allRegions()]
  .filter(r => r.section === 'persistent-skills' || r.section === 'session-skills')
  .map(r => (r.content as { name: string }).name),
contextEpoch: this.regions.getEpoch(),
```

- [ ] **Step 4: Verify unit tests still pass**

Run: `npm run test:unit`
Expected: 30 pass (unit tests use mocked LLM; assemble produces different bytes but test assertions don't compare those bytes)

- [ ] **Step 5: Verify deterministic e2e — expect BREAKAGE**

Run: `npm run test:e2e:deterministic`
Expected: **FAIL** in `tests/e2e/deterministic.test.ts` and/or `s-011-*.test.ts` due to replay divergence — the recorded request's `system` field bytes no longer match the new assembled output.

That's expected. Fixture re-record happens in Task 12.

**Acceptance:** The failure mode is `ReplayDivergenceError` (or similar) at request comparison, NOT a structural error in assemble.

- [ ] **Step 6: Commit**

```bash
git add src/runtime/AgentRuntime.ts
git commit -m "feat(runtime): use assemble for LLM request (PR-C1 step 7/13 — e2e BREAK expected)"
```

---

## Task 8: AgentRuntime — migrate skill loading to region inserts

**Files:**
- Modify: `src/runtime/AgentRuntime.ts`

**Why:** `pendingSkills: Set<string>` + `applyPendingSkills` writes to old `ContextLayer.loadInstructions`. Switch to writing skill regions directly.

- [ ] **Step 1: Replace pendingSkills field + applyPendingSkills**

Replace line 56:

```typescript
private pendingSkills: Set<string> = new Set()
```

with:

```typescript
private pendingSkillLoads: Array<{ name: string; instructions: string }> = []
```

Replace `requestSkill` (lines 227-236):

```typescript
private requestSkill(name: string): { requested: string; status: string; version?: string } {
  const normalized = name.trim().replace(/\s+skill$/i, '')
  const version = this.config.skills?.[normalized]
  const instructions = this.config.skillInstructions?.[normalized]
  if (!version || !instructions) {
    return { requested: name, status: 'unavailable' }
  }
  this.pendingSkillLoads.push({ name: normalized, instructions })
  return { requested: normalized, status: 'pending_next_epoch', version }
}
```

Replace `applyPendingSkills` (lines 238-246):

```typescript
private applyPendingSkills(): void {
  for (const { name, instructions } of this.pendingSkillLoads) {
    const id = `skill:${name}`
    // Already loaded? Skip (preserves createdAt; idempotent like Map.set upsert).
    if (this.regions.get(id)) continue
    // PR-C1 default: session-scope (matches old ContextLayer behavior).
    // PR-C2 will introduce turn-scope as the new default + plumb scope through requestSkill.
    this.regions.set(id, makeSkillRegion(name, instructions, 'session'))
    // Keep ContextLayer in sync until Task 11 deletion.
    this.context.loadInstructions(name, instructions)
  }
  this.pendingSkillLoads = []
}
```

- [ ] **Step 2: Verify unit tests pass**

Run: `npm run test:unit`
Expected: 30 pass

- [ ] **Step 3: Commit**

```bash
git add src/runtime/AgentRuntime.ts
git commit -m "feat(runtime): skill loads route through ContextRegions (PR-C1 step 8/13)"
```

---

## Task 9: AgentRuntime — wire inter-turn engine at run/continueTurn boundaries

**Files:**
- Modify: `src/runtime/AgentRuntime.ts`

**Why:** Crystallization fires when a turn completes. "Completes" means `executeFSM` returned without throwing AND we're not interrupted.

- [ ] **Step 1: Add import**

```typescript
import { runInterTurnEngine } from '../context/lifecycleEngine.js'
```

- [ ] **Step 2: Add private helper**

```typescript
private crystallizeTurn(userInput: string): void {
  runInterTurnEngine(this.regions, {
    boundary:  'turn-end',
    userInput,
    now:       this.ioPort.now(),
  })
  // Mirror the cross-turn effect on ContextLayer.history (kept in sync for
  // PR-C1; Task 11 deletion removes this duplication).
  // Old ContextLayer kept assistant + tool messages flat; the new model
  // collapses them to one (user, finalAssistant) pair. To avoid history
  // divergence between mirror and source for the remaining ContextLayer
  // consumer (Replay.nondet.test.ts), we don't try to mirror — Task 12
  // rewrites that test.
}
```

- [ ] **Step 3: Call crystallizeTurn at run/continueTurn end**

In `run` (line 318 area), wrap the existing logic:

```typescript
async run(input: string): Promise<AgentResult> {
  this.rootSpan = this.recorder.startSpan('agent.run', {
    agentId:   this.config.agentId,
    goal:      this.goal,
    contextId: this.contextId,
  })

  const turnInput = `Goal: ${this.goal}\n\n${input}`
  this.setCurrentTurn(turnInput)
  this.turnNumber++

  try {
    await this.executeFSM()
    // Turn completed successfully — crystallize.
    this.crystallizeTurn(turnInput)
    this.recorder.endSpan(this.rootSpan, 'ok')
    return {
      agentRunId: this.agentRunId,
      contextId:  this.contextId,
      output:     this.lastTextOutput,
      status:     'completed',
    }
  } catch (err: unknown) {
    // Note: do NOT crystallize on interrupt — partial turn state stays
    // in scratchpad until resume; crystallization happens on the eventual
    // successful turn completion.
    const isInterrupt = err instanceof Error && err.name === 'InterruptSignal'
    if (isInterrupt) { /* unchanged */ }
    /* unchanged error path */
  } finally {
    await this.recorder.flush()
  }
}
```

In `continueTurn` (line 362), update similarly so the new `turnInput` is used:

```typescript
async continueTurn(input: string): Promise<AgentResult> {
  this.setCurrentTurn(input)
  this.turnNumber++
  return this.run(input)  // unchanged — run already does crystallize on success
}
```

Wait — `continueTurn` calls `run(input)` which internally re-sets `setCurrentTurn(\`Goal: ${this.goal}\n\n${input}\`)` overwriting the simpler input. Verify that's the existing behavior (it is — re-read line 325). Keep behavior intact.

- [ ] **Step 4: Verify unit tests pass**

Run: `npm run test:unit`
Expected: 30 pass

- [ ] **Step 5: Commit**

```bash
git add src/runtime/AgentRuntime.ts
git commit -m "feat(runtime): crystallize turn on run/continueTurn completion (PR-C1 step 9/13)"
```

---

## Task 10: AgentRuntime — migrate checkpoint save/restore to regions snapshot

**Files:**
- Modify: `src/runtime/AgentRuntime.ts`
- Modify: `src/types/store.ts` (if checkpoint type shape changes)

**Why:** Checkpoint must survive process restart for multi-turn continuation. Today's checkpoint stores `context.{history, instructionsSnapshot, instructions, contextEpoch}`. New shape stores `context.regions` (the RegionSnapshot blob from `regions.snapshot()`).

**Compatibility:** This is a **breaking change to checkpoint format**. Per spec §9.1 we're not preserving byte-identical fixtures. Old checkpoints can't be loaded by the new code — acceptable per "未上线无需向后兼容".

- [ ] **Step 1: Check current AgentCheckpoint shape**

Read `src/types/store.ts`. Find `AgentCheckpoint` interface and the `context` field.

- [ ] **Step 2: Add `regions` field to AgentCheckpoint.context**

In `src/types/store.ts`, modify the `context` field of `AgentCheckpoint`:

```typescript
context: {
  workingMemory:        unknown   // existing
  regions:              import('../context/Region.js').RegionSnapshot
  // Deprecated (kept readable for older fixtures during this PR; Task 12 deletes):
  history?:             Message[]
  instructionsSnapshot?: string[]
  instructions?:        Record<string, string>
  contextEpoch?:        number
}
```

(Pull in `Message` import if not already present.)

- [ ] **Step 3: Update saveCheckpoint (line 274 area)**

```typescript
private async saveCheckpoint(resumeState?: string, currentTurn?: string): Promise<AgentCheckpoint> {
  return this.checkpoints.save({
    sequence:    this.turnNumber,
    goal:        this.goal,
    currentTurn: currentTurn ?? this.getCurrentTurn() ?? undefined,
    fsm:         this.fsm.snapshot(resumeState),
    context: {
      workingMemory: this.memory.toJSON(),
      regions:       this.regions.snapshot(),
    },
    pendingEvents: this.pendingEvents.map(e => ({ type: e.type, payload: e.payload })),
    children:      Array.from(this.childRecords.values()),
    meta: {
      agentId:       this.config.agentId,
      agentRunId:    this.agentRunId,
      parentAgentId: this.parentId,
      timestamp:     this.ioPort.now(),
      traceId:       (this.recorder as Partial<InMemoryRecorder>).traceId ?? '',
      contextId:      this.contextId,
    },
  })
}
```

- [ ] **Step 4: Update loadCheckpoint (line 302 area)**

```typescript
async loadCheckpoint(checkpoint: AgentCheckpoint): Promise<void> {
  this.turnNumber = checkpoint.sequence
  this.regions.restore(checkpoint.context.regions)
  const restoredMemory = WorkingMemory.fromJSON(checkpoint.context.workingMemory)
  Object.assign(this.memory, restoredMemory)
  this.fsm.restore(checkpoint.fsm)
  if (checkpoint.fsm.currentState === 'paused' && checkpoint.fsm.resumeState) {
    this.fsm.transitionTo(checkpoint.fsm.resumeState, { name: 'RESUME' })
  }
}
```

- [ ] **Step 5: Verify**

Run: `npm run test:unit`
Expected: 30 pass

- [ ] **Step 6: Commit**

```bash
git add src/runtime/AgentRuntime.ts src/types/store.ts
git commit -m "feat(runtime): checkpoint stores regions snapshot (PR-C1 step 10/13 — BREAKING checkpoint format)"
```

---

## Task 11: Delete ContextLayer

**Files:**
- Delete: `src/context/ContextLayer.ts`
- Modify: `src/runtime/AgentRuntime.ts` — remove ContextLayer field + import + all remaining sync calls
- Modify: `src/index.ts` — remove ContextLayer re-export

**Why:** Now safe to delete. Every consumer either uses ContextRegions or will be rewritten in Task 12.

- [ ] **Step 1: Remove ContextLayer usages from AgentRuntime**

In `src/runtime/AgentRuntime.ts`:

- Remove import `import { ContextLayer } from '../context/ContextLayer.js'`
- Remove field `private readonly context: ContextLayer`
- Remove constructor block `this.context = new ContextLayer({...})`
- In `setCurrentTurn`: remove `this.context.setCurrentTurn(input)` mirror call
- In `appendScratchpadAssistant`: remove `this.context.appendHistory(...)` mirror call
- In `appendScratchpadToolResults`: remove `this.context.appendHistory(...)` mirror call
- In `applyPendingSkills`: remove `this.context.loadInstructions(...)` mirror call

- [ ] **Step 2: Remove ContextLayer re-export**

Check `src/index.ts`:

```bash
grep ContextLayer src/index.ts
```

If found, remove the line.

- [ ] **Step 3: Delete the file**

```bash
git rm src/context/ContextLayer.ts
```

- [ ] **Step 4: TypeScript check should fail on Replay.nondet.test.ts (fixed in Task 12)**

Run: `npm run test:unit`
Expected: FAIL — `Replay.nondet.test.ts` imports `ContextLayer`.

Run: `npm run test:unit -- --testPathIgnorePatterns=Replay.nondet`
Expected: 24 tests pass (the remaining suites).

- [ ] **Step 5: Commit**

```bash
git add src/runtime/AgentRuntime.ts src/context/ContextLayer.ts src/index.ts
git commit -m "feat(context): delete ContextLayer (PR-C1 step 11/13 — Replay.nondet.test broken until step 12)"
```

---

## Task 12: Update Replay.nondet.test + re-record versionable fixtures

**Files:**
- Modify: `src/__tests__/Replay.nondet.test.ts`
- Re-record: `examples/s-005-replay/.milkie/runs/8da9174a-567d-406a-9596-9ee53947b194.jsonl`
- Re-record: `examples/s-002-inspect/.milkie/runs/1e65a3ec-03d4-40b4-9546-983fce9cb0e3.jsonl`

**Why:** Old test references ContextLayer directly. Fixtures contain old assembled-system bytes that no longer match the new substrate's output.

- [ ] **Step 1: Read Replay.nondet.test.ts to understand what it asserts**

Run:

```bash
wc -l src/__tests__/Replay.nondet.test.ts
grep -n "ContextLayer" src/__tests__/Replay.nondet.test.ts
```

Then read the file fully.

- [ ] **Step 2: Rewrite to use ContextRegions**

The test exercises ContextLayer's snapshot/restore + non-determinism replay. Rewrite to use `ContextRegions.snapshot()/restore()` directly. If the test's assertions are still about ContextLayer-specific behavior that has no analog in the new substrate (e.g. "instructionsSnapshot field present"), delete those assertions. Replace with the equivalent ContextRegions invariants (e.g. "after restore, get('skill:verifier') returns the right region").

(Exact code depends on the file contents read in Step 1.)

- [ ] **Step 3: Run test**

```bash
npx jest src/__tests__/Replay.nondet.test.ts
```

Expected: PASS.

- [ ] **Step 4: Identify fixture re-record commands**

Look for "record" mode invocation in `examples/s-005-replay/` and `examples/s-002-inspect/`:

```bash
cat examples/s-005-replay/package.json examples/s-005-replay/README.md 2>/dev/null | head -50
cat examples/s-002-inspect/package.json examples/s-002-inspect/README.md 2>/dev/null | head -50
```

These examples should have a `record.ts` or similar; record mode regenerates the .jsonl.

- [ ] **Step 5: Re-record both fixtures**

For each example:

```bash
cd examples/s-005-replay
# Delete the old fixture
rm .milkie/runs/*.jsonl
# Re-record
npx tsx record.ts   # or whatever the record script is — adjust per Step 4 findings
# Verify a new .jsonl was created
ls .milkie/runs/
```

Repeat for `examples/s-002-inspect`.

- [ ] **Step 6: Re-run full e2e suite**

```bash
npm run test:e2e:deterministic
```

Expected: 7 pass + 15 skipped. If any FAIL, the fixture re-record may have generated a different UUID/filename — update the test's hardcoded path.

- [ ] **Step 7: Commit**

```bash
git add src/__tests__/Replay.nondet.test.ts examples/s-005-replay/.milkie examples/s-002-inspect/.milkie
git commit -m "test(replay): rewrite Replay.nondet + re-record versionable fixtures (PR-C1 step 12/13)"
```

---

## Task 13: Full suite + PR

**Files:** none

- [ ] **Step 1: Run full test suite**

```bash
npm run test:unit && npm run test:e2e:deterministic
```

Expected: 30 + 7 pass, no skips beyond the 15 known-skipped non-deterministic ones.

- [ ] **Step 2: Confirm no orphan files**

```bash
grep -rn "ContextLayer" src/ examples/ tests/ 2>/dev/null
```

Expected: zero matches. If matches found, investigate (probably a missed import).

- [ ] **Step 3: Push branch**

```bash
git push -u origin feat/context-regions-substrate-pr-c1
```

- [ ] **Step 4: Open PR**

```bash
gh pr create --title "feat(runtime): substrate rewrite — AgentRuntime on ContextRegions+assemble (PR-C1)" --body "<filled in at execution time>"
```

PR body must include:
- Reference to spec sections covered (§4.2–§4.4, §5, §7, §11)
- List of breaking changes: replay fixture re-recorded, checkpoint format incompatible with pre-PR-C1 snapshots
- Test summary (unit + e2e counts, lifecycleEngine test count)
- What's NOT in (PR-C2 skill scope param, PR-D trace events / cache health, PR-C3 tool result strategy deferred)
- Honest note that any non-deterministic e2e tests (`test:e2e:live`) need re-recording too — not done in this PR, tracked separately

---

## Spec coverage self-check

| Spec section | Where it lands |
|---|---|
| §4.2 ContextRegions class | PR-A (merged) + Task 4 here adds the field to AgentRuntime |
| §4.3 skill lifetime model | **Partial** — Task 8 adds the region path; full `scope: 'turn'|'session'` parameterization is **PR-C2** |
| §4.4 ToolResultStrategy | **Deferred to PR-C3 or later** (user-approved per chat — example already README-flagged) |
| §5 assemble function | PR-B (merged) + Task 7 here calls it |
| §6 cache-aware section schema | PR-B (merged) — no runtime work needed in PR-C1 |
| §7.1 boundaries diagram | Task 9 wires turn boundary |
| §7.2 intra-turn engine | **Deferred** — Phase 1 implementation is the existing FSM loop; explicit `runIntraTurnEngine` becomes useful only when state-scoped expiry / tool-buffer policies are exercised by real agents. Tracked in PR-C2 or later. |
| §7.3 inter-turn engine | Task 3 implements; Task 9 wires |
| §7.4 hook points | Task 7 (assemble) + Task 9 (crystallize) |
| §11 invariants | Items 1, 2, 4, 5, 7, 9 covered by PR-A/PR-B unit tests + Task 3 lifecycleEngine tests. Items 3, 6, 8 deferred to PR-C2 / PR-D. |

**Scope changes from spec:**
1. `runIntraTurnEngine` not implemented in PR-C1 — added when the lifecycle states it manages (`state-scoped` expiry, `tool-buffer`, `one-shot`) are actually used. Currently the FSM loop naturally invalidates state-scoped regions through `isActive` filtering in assemble (PR-B). Belt without suspenders is acceptable for Phase 1.
2. Skill scope parameter (`turn`/`session`) defaults to `'session'` in PR-C1 to match old ContextLayer behavior — PR-C2 introduces the parameter at the tool surface.
3. Single-tool ToolResultStrategy deferred — see PR-C3 / Phase 5.
