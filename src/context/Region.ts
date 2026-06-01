// Region type definitions for the context substrate.
// Spec: docs/superpowers/specs/2026-05-25-context-region-substrate-design.md §4.1

export type RegionTarget = 'system' | 'message' | 'tool'

export type SystemSection =
  | 'header'
  | 'persistent-skills'
  | 'tools-static'
  | 'session-skills'
  | 'state'
  | 'tools-state'
  | 'wm'
  | 'footer'

export type MessageSection =
  | 'history'
  | 'turn-context'   // #82: per-turn injected variables (volatile, between history and current-turn)
  | 'current-turn'
  | 'scratchpad'

export type ToolSection = 'default'

export type RegionSection = SystemSection | MessageSection | ToolSection

export type IntraTurnScope =
  | 'turn-persistent'
  | { kind: 'state-scoped'; state: string }
  | { kind: 'tool-buffer'; remainingCalls: number }
  | 'one-shot'

export type InterTurnScope =
  | 'session-persistent'
  | 'turn-local'
  | { kind: 'ttl'; deadline: number }
  | 'summarize-on-overflow'
  | 'promote-to-wm'

export type RegionStability =
  | 'immutable'
  | 'session-stable'
  | 'turn-stable'
  | 'volatile'

// Format functions return one of three shapes depending on target:
//   target='system' → string (rendered into system prompt block)
//   target='message' → Message or Message[] (folded into messages array)
//   target='tool'   → ToolSchema (folded into tools array)
// Concrete Message / ToolSchema types live elsewhere; Region itself stays
// substrate-level and does not import them — the format function's caller
// (assemble in PR-B) is responsible for typing the output correctly.
export type RegionFormatOutput = string | object | ReadonlyArray<object>

export interface Region {
  readonly id:         string
  readonly target:     RegionTarget
  readonly section:    RegionSection

  readonly ordinal?:   number
  readonly createdAt:  number

  readonly intraTurn:  IntraTurnScope
  readonly interTurn:  InterTurnScope

  readonly stability:       RegionStability
  readonly cacheBreakpoint?: boolean

  readonly content: unknown
  readonly format:  (content: unknown) => RegionFormatOutput
}

// Input shape for ContextRegions.set — id and createdAt are filled by the
// store (id from the caller's argument, createdAt from the injected clock).
export type RegionInput = Omit<Region, 'id' | 'createdAt'>

// Snapshot for checkpoint / restore. Format functions are not serializable;
// the consumer (checkpoint loader) re-attaches them by region id or section.
// PR-A keeps the snapshot opaque; PR-C will refine when restore semantics
// concretize.
export interface RegionSnapshot {
  readonly epoch:   number
  readonly regions: ReadonlyArray<Region>
}
