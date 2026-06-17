// Scenario-layer user-facing wording (#180) for repair-ticketing.
//
// The portable core (resolver) returns only neutral status + structured data —
// never话术. This module renders the Chinese the END USER sees, keyed off those
// statuses/outcomes. Model-facing话术 (system prompt, tool descriptions) lives in
// agent.ts; keep these two separate so wording changes never touch logic/core.

/** A candidate as surfaced to the user during clarification. */
export interface ChoiceLabel {
  id: string
  label: string
  path: string[]
}

const fmtChoices = (cands: ChoiceLabel[]): string =>
  cands.map(c => `${c.id}（${c.label}）`).join('、')

export const repairMessages = {
  /** Multiple candidates at a level and context can't disambiguate → ask the user. */
  clarifyAmbiguous: (levelLabel: string, cands: ChoiceLabel[]): string =>
    `「${levelLabel}」匹配到多个，请确认是哪一个：${fmtChoices(cands)}。`,

  /** The next level isn't yet resolvable from this turn — prompt for it. Covers both
   *  "the user hasn't named it yet" and "what they said didn't match anything". */
  promptForLevel: (levelLabel: string): string =>
    `请提供报修对象的${levelLabel}（名称、别名或简称均可）。`,

  /** A selection conflicted with the confirmed branch and could not be corrected. */
  rejectInvalid: (levelLabel: string): string =>
    `该${levelLabel}与已确认的上级不一致，无法采用，请重新提供。`,

  /** A selection was auto-corrected to stay consistent with the confirmed branch. */
  noticeCorrected: (levelLabel: string, toLabel: string): string =>
    `已按已确认的分支将${levelLabel}校正为「${toLabel}」。`,
}

export type RepairMessages = typeof repairMessages
