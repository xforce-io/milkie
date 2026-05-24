/**
 * Thrown when the replayed agent issues an LLM or tool call whose
 * canonical hash does not appear (or appears too few times) in the
 * recorded event log. Replay is strict — divergence is fail-fast.
 */
export class ReplayDivergenceError extends Error {
  constructor(
    public readonly kind:            'llm' | 'tool',
    public readonly actualHash:      string,
    public readonly summary:         string,
    public readonly availableHashes: string[],
  ) {
    super(`Replay divergence (${kind}): hash ${actualHash.slice(0, 12)}… not in cache. ${summary}`)
    this.name = 'ReplayDivergenceError'
  }
}
