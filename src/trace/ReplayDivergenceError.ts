export type DivergenceKind = 'llm' | 'tool' | 'clock' | 'uuid'

/**
 * Thrown when replay diverges from the recorded log:
 *  - llm/tool: replayed call's canonical hash not in cache (over-consume)
 *  - clock/uuid: replay called port.now/port.uuid more times than recorded
 *    OR replay completed with unconsumed recorded values (under-consume)
 *
 * For llm/tool, actualHash + availableHashes carry the hash diagnostic;
 * for clock/uuid the message itself carries the count, hash fields are
 * empty placeholders.
 */
export class ReplayDivergenceError extends Error {
  constructor(
    public readonly kind:            DivergenceKind,
    public readonly actualHash:      string,
    public readonly summary:         string,
    public readonly availableHashes: string[],
  ) {
    const detail = (kind === 'llm' || kind === 'tool')
      ? `hash ${actualHash.slice(0, 12)}… not in cache. ${summary}`
      : summary
    super(`Replay divergence (${kind}): ${detail}`)
    this.name = 'ReplayDivergenceError'
  }
}
