// Section ordering schema for assemble.
// Spec: docs/superpowers/specs/2026-05-25-context-region-substrate-design.md §5 + §6
//
// Sections are ordered by stability (most stable first) so that prefix cache
// hits the longest stable prefix. Cache breakpoint candidates are marked.

import type { RegionTarget, MessageSection, SystemSection, ToolSection } from './Region'

export const SECTION_SCHEMA: {
  readonly system:  ReadonlyArray<SystemSection>
  readonly message: ReadonlyArray<MessageSection>
  readonly tool:    ReadonlyArray<ToolSection>
} = {
  system: [
    'header',
    'persistent-skills',
    'tools-static',
    // ─── cache breakpoint candidate: stable cut ───
    'session-skills',
    // ─── cache breakpoint candidate: session cut ───
    'state',
    'tools-state',
    'wm',
    // ─── cache breakpoint candidate: turn cut ───
    'footer',
  ],
  message: [
    'history',
    'current-turn',
    'scratchpad',
  ],
  tool: ['default'],
}

export function sectionsFor(target: RegionTarget): ReadonlyArray<string> {
  return SECTION_SCHEMA[target]
}
