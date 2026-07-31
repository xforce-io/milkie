# Discover Tool Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runtime-enforced optional per-agent tool-call budget for Researcher #117.

**Architecture:** Agent frontmatter supplies `fsm.max_tool_calls`; each AgentRuntime run owns one shared counter. Before every serial or parallel handler dispatch it reserves a slot; calls without a slot emit `TOOL_CALL_BUDGET_EXCEEDED` through the existing tool error/trace path.

**Tech Stack:** TypeScript, Zod/frontmatter config, AgentRuntime, Jest.

## Global Constraints

- Omitted budget preserves unlimited historical behavior.
- Rejected calls never execute handlers.
- Parallel calls share one per-run budget.

---

### Task 1: Parse and validate max_tool_calls

**Files:**
- Modify: `src/agent.ts` and the existing agent config type/schema source
- Test: `src/__tests__/AgentRuntime.test.ts`

- [ ] **Step 1: Add failing config cases**

```ts
expect(loadAgent(validFrontmatter).fsm.max_tool_calls).toBe(2);
expect(() => loadAgent('max_tool_calls: -1')).toThrow();
```

- [ ] **Step 2: Run focused test**

Run: `npm run test:unit -- --runInBand src/__tests__/AgentRuntime.test.ts`
Expected: field is absent or not validated.

- [ ] **Step 3: Add optional non-negative integer config field**

```ts
max_tool_calls?: number;
```

- [ ] **Step 4: Re-run focused test**

Run: `npm run test:unit -- --runInBand src/__tests__/AgentRuntime.test.ts`
Expected: valid budgets load and invalid budgets reject.

### Task 2: Enforce the shared dispatch budget

**Files:**
- Modify: `src/runtime/AgentRuntime.ts`, `src/runtime/IOPort.ts`, `src/trace/RecordingIOPort.ts`
- Test: `src/__tests__/AgentRuntime.test.ts`, `src/__tests__/AgentRuntime.toolResultStrategy.test.ts`, `src/__tests__/RecordingIOPort.toolCallId.test.ts`

- [ ] **Step 1: Add failing serial and parallel tests**

```ts
expect(executed).toEqual(['first', 'second']);
expect(rejected.error.code).toBe('TOOL_CALL_BUDGET_EXCEEDED');
expect(parallelHandler).toHaveBeenCalledTimes(2);
```

- [ ] **Step 2: Run focused runtime tests**

Run: `npm run test:unit -- --runInBand src/__tests__/AgentRuntime.test.ts src/__tests__/AgentRuntime.toolResultStrategy.test.ts src/__tests__/RecordingIOPort.toolCallId.test.ts`
Expected: all calls execute before the budget gate exists.

- [ ] **Step 3: Reserve a slot before handler invocation**

```ts
private tryConsumeToolCall(): boolean {
  if (this.maxToolCalls === undefined) return true;
  if (this.toolCallsUsed >= this.maxToolCalls) return false;
  this.toolCallsUsed += 1;
  return true;
}
```

Map failed reservations to an existing traceable error ToolResult; do not call the handler thunk.

- [ ] **Step 4: Re-run focused runtime tests**

Run: `npm run test:unit -- --runInBand src/__tests__/AgentRuntime.test.ts src/__tests__/AgentRuntime.toolResultStrategy.test.ts src/__tests__/RecordingIOPort.toolCallId.test.ts`
Expected: serial and parallel caps are deterministic and rejected handlers are never called.

### Task 3: Complete regression verification and commit

**Files:**
- Test: `tests/e2e/s-001-react-with-intra-agent-parallel-tools.e2e.test.ts`

- [ ] **Step 1: Add a bounded parallel e2e case**

```ts
expect(trace.toolRequested).toHaveLength(2);
expect(trace.toolResponded.at(-1)?.error.code).toBe('TOOL_CALL_BUDGET_EXCEEDED');
```

- [ ] **Step 2: Run all validation**

Run: `npm test -- --runInBand`
Expected: all suites pass.

- [ ] **Step 3: Commit**

```bash
git add src src/__tests__ tests/e2e docs/superpowers/plans/2026-07-31-discover-tool-budget.md
git commit -m "feat: enforce agent tool call budgets"
```
