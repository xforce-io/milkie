// Deterministic multi-algorithm fusion recall (#180) — the portable, LLM-free
// "step 1" of repair-ticketing's slot filling.
//
// Given an utterance + already-pinned ancestors, recall scores every entity at
// each remaining level with several independent matchers (exact / substring /
// char-n-gram / edit distance), each producing its own ranking, then fuses the
// rankings with Reciprocal Rank Fusion (RRF). The output is NEUTRAL data —
// candidate ids, scores, paths, which matcher hit, and a `decisive` fast-path
// hint — with NO user-facing wording and NO LLM. Scenario话术 and the LLM
// final-select (step 2) live in the adapter layer (#166), not here.
//
// Zero new dependencies by design (#167 portability). Pinyin and full BM25-idf
// weighting are deliberately out of this first cut (pinyin needs a hanzi→pinyin
// table = a dependency); the four matchers below already cover exact / alias /
// substring / partial / typo.

import type { HierarchicalDict, EntityRecord } from './EntityResolver.js'
import { targetLevels, ancestorOnly, matchesPinned } from './EntityResolver.js'

export type MatchAlgo = 'exact' | 'substring' | 'ngram' | 'edit'

/** A neutral recall candidate — real entity id the LLM may later pick from. */
export interface RecallCandidate {
  id: string
  label: string
  level: string
  path: string[]          // full ancestor path → tree topology for step-2 judging
  score: number           // fused RRF score (higher = better)
  via: MatchAlgo[]        // which matchers contributed (descending contribution)
  matchedSurface: string  // the label/alias surface form that matched best
}

export interface LevelRecall {
  level: string
  candidates: RecallCandidate[]   // pinned-filtered, score-descending, capped to topK
  decisive: string | null         // fast-path: a clearly-unique winner's id, else null
}

export interface RecallResult {
  byLevel: LevelRecall[]          // remaining (unfilled) levels only
}

export interface RecallConfig {
  algos?: MatchAlgo[]             // default: all four
  rrfK?: number                   // RRF constant, default 60
  topK?: number                   // candidates kept per level, default 5
  ngram?: number                  // char n-gram size for ngram/edit windows, default 2
  minScore?: number               // drop candidates scoring below this (post-fusion), default 0
  decisiveRatio?: number          // top must beat 2nd by this RRF ratio to be decisive, default 1.5
}

const DEFAULTS: Required<RecallConfig> = {
  algos: ['exact', 'substring', 'ngram', 'edit'],
  rrfK: 60,
  topK: 5,
  ngram: 2,
  minScore: 0,
  decisiveRatio: 1.5,
}

// ─── per-matcher scoring (surface form vs utterance) ──────────────────────────
// Each returns 0..1. The utterance is matched against an entity's surface forms
// (label + alias columns); the entity's per-matcher score is its best surface.

function scoreExact(surface: string, q: string): number {
  if (!surface) return 0
  if (surface === q) return 1
  // The whole entity name appears verbatim inside a longer utterance ("…网络部…").
  if (q.includes(surface)) return 0.95
  return 0
}

function scoreSubstring(surface: string, q: string): number {
  if (!surface) return 0
  if (q.includes(surface)) return Math.min(1, surface.length / Math.max(1, q.length)) * 0.6 + 0.4
  if (surface.includes(q)) return Math.min(1, q.length / Math.max(1, surface.length)) * 0.6 + 0.3
  return 0
}

/** Char n-gram overlap (Dice coefficient) — BM25-style lexical recall for short
 *  strings, dependency-free. Catches reordering / partial overlap. */
function scoreNgram(surface: string, q: string, n: number): number {
  const a = charNgrams(surface, n)
  const b = charNgrams(q, n)
  if (a.size === 0 || b.size === 0) return 0
  let inter = 0
  for (const g of a) if (b.has(g)) inter++
  return (2 * inter) / (a.size + b.size)
}

/** Best normalized edit-distance similarity of `surface` against any same-length
 *  window of the utterance — typo tolerance ("总布"→"总部") without matching a
 *  short name against the whole long sentence.
 *
 *  First-char anchor: only windows that share the surface's first character count.
 *  Short Chinese names make raw edit distance noisy — a window like "络部" is one
 *  edit from "总部" purely by a coincidental tail char, the same 0.5 similarity a
 *  real typo "总布" scores. Anchoring on the first char keeps real typos (which
 *  rarely corrupt the leading char) and drops tail-coincidence noise. The rare
 *  leading-char typo is given up here — substring/ngram/step-2 can still catch it. */
function scoreEdit(surface: string, q: string): number {
  if (!surface) return 0
  const w = surface.length
  const head = surface[0]
  if (q.length <= w) return q[0] === head ? editSim(surface, q) : 0
  let best = 0
  for (let i = 0; i + w <= q.length; i++) {
    if (q[i] !== head) continue
    best = Math.max(best, editSim(surface, q.slice(i, i + w)))
    if (best === 1) break
  }
  return best
}

// ─── primitives ───────────────────────────────────────────────────────────────

function charNgrams(s: string, n: number): Set<string> {
  const out = new Set<string>()
  if (s.length < n) { if (s) out.add(s); return out }
  for (let i = 0; i + n <= s.length; i++) out.add(s.slice(i, i + n))
  return out
}

function editSim(a: string, b: string): number {
  const d = levenshtein(a, b)
  const m = Math.max(a.length, b.length)
  return m === 0 ? 0 : 1 - d / m
}

function levenshtein(a: string, b: string): number {
  const m = a.length, n = b.length
  if (m === 0) return n
  if (n === 0) return m
  let prev = Array.from({ length: n + 1 }, (_, j) => j)
  let cur = new Array<number>(n + 1)
  for (let i = 1; i <= m; i++) {
    cur[0] = i
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      cur[j] = Math.min(prev[j]! + 1, cur[j - 1]! + 1, prev[j - 1]! + cost)
    }
    [prev, cur] = [cur, prev]
  }
  return prev[n]!
}

const SCORERS: Record<MatchAlgo, (surface: string, q: string, n: number) => number> = {
  exact:     (s, q) => scoreExact(s, q),
  substring: (s, q) => scoreSubstring(s, q),
  ngram:     (s, q, n) => scoreNgram(s, q, n),
  edit:      (s, q) => scoreEdit(s, q),
}

// ─── per-level recall ──────────────────────────────────────────────────────────

interface Scored {
  entity: EntityRecord
  perAlgo: Map<MatchAlgo, number>   // best surface score per matcher
  bestSurface: Map<MatchAlgo, string>
}

function scoreLevel(
  entities: EntityRecord[],
  q: string,
  cfg: Required<RecallConfig>,
): Scored[] {
  const out: Scored[] = []
  for (const entity of entities) {
    const surfaces = [entity.label, ...Object.values(entity.searchValues)]
      .map(s => (s ?? '').toLowerCase())
      .filter(Boolean)
    const perAlgo = new Map<MatchAlgo, number>()
    const bestSurface = new Map<MatchAlgo, string>()
    for (const algo of cfg.algos) {
      let best = 0, bestS = ''
      for (const surface of surfaces) {
        const s = SCORERS[algo](surface, q, cfg.ngram)
        if (s > best) { best = s; bestS = surface }
      }
      if (best > 0) { perAlgo.set(algo, best); bestSurface.set(algo, bestS) }
    }
    if (perAlgo.size > 0) out.push({ entity, perAlgo, bestSurface })
  }
  return out
}

/** Fuse per-matcher rankings into one score via Reciprocal Rank Fusion. */
function fuse(scored: Scored[], cfg: Required<RecallConfig>): Map<string, number> {
  const fused = new Map<string, number>()
  for (const algo of cfg.algos) {
    const ranked = scored
      .filter(s => s.perAlgo.has(algo))
      .sort((a, b) => b.perAlgo.get(algo)! - a.perAlgo.get(algo)!)
    ranked.forEach((s, i) => {
      fused.set(s.entity.id, (fused.get(s.entity.id) ?? 0) + 1 / (cfg.rrfK + i + 1))
    })
  }
  return fused
}

function decisiveOf(cands: RecallCandidate[], cfg: Required<RecallConfig>): string | null {
  if (cands.length === 0) return null
  if (cands.length === 1) return cands[0]!.id
  const [a, b] = cands
  const strong = (c: RecallCandidate) => c.via.includes('exact') || c.via.includes('substring')
  // A single strong direct hit with no equally-strong rival → accept.
  if (strong(a!) && !strong(b!)) return a!.id
  // Otherwise require a clear RRF lead.
  if (a!.score >= b!.score * cfg.decisiveRatio) return a!.id
  return null
}

/**
 * Recall candidates for every remaining level, fused and ranked. Pure, deterministic,
 * LLM-free. `pinned` filters to the matching branch and (level-less) derives which
 * levels to search; `level` overrides to a single level.
 */
export function recall(
  dict: HierarchicalDict,
  utterance: string,
  pinned: Record<string, string> = {},
  config: RecallConfig = {},
  level?: string,
): RecallResult {
  const cfg = { ...DEFAULTS, ...config }
  const q = utterance.toLowerCase().trim()
  const byLevel: LevelRecall[] = []

  for (const lvl of targetLevels(dict, pinned, level)) {
    const levelMap = dict.index.get(lvl)
    if (!levelMap) continue
    const ancestorPins = ancestorOnly(pinned, lvl)
    const entities = [...levelMap.values()].filter(e => matchesPinned(e, ancestorPins))

    const scored = q ? scoreLevel(entities, q, cfg) : []
    const fused = fuse(scored, cfg)

    let candidates: RecallCandidate[] = scored.map(s => {
      const via = [...s.perAlgo.entries()].sort((a, b) => b[1] - a[1]).map(([algo]) => algo)
      return {
        id: s.entity.id,
        label: s.entity.label,
        level: lvl,
        path: s.entity.path,
        score: fused.get(s.entity.id) ?? 0,
        via,
        matchedSurface: s.bestSurface.get(via[0]!) ?? s.entity.label,
      }
    })
    candidates = candidates
      .filter(c => c.score > cfg.minScore)
      .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
      .slice(0, cfg.topK)

    byLevel.push({ level: lvl, candidates, decisive: decisiveOf(candidates, cfg) })
  }

  return { byLevel }
}
