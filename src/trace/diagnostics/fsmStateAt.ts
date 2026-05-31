import type { Event, FsmTransitionPayload } from '../types.js'

/**
 * The FSM state in effect at `eventId`: fold fsm.transition events strictly
 * before it and take the last `to`. If none precede it, the state is the
 * initial state — revealed by the `from` of the run's first transition.
 * Returns null when the run has no transitions at all. Pure.
 */
export function fsmStateAt(events: Event[], eventId: string): string | null {
  const index = events.findIndex(e => e.id === eventId)
  const cut = index < 0 ? events.length : index
  let state: string | null = null
  for (let i = 0; i < cut; i++) {
    const e = events[i]!
    if (e.type === 'fsm.transition') state = (e.payload as FsmTransitionPayload).to
  }
  if (state !== null) return state
  const first = events.find(e => e.type === 'fsm.transition')
  return first ? (first.payload as FsmTransitionPayload).from : null
}
