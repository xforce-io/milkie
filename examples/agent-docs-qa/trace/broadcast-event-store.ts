// #86: BroadcastingEventStore was promoted into the framework (src/trace/) so
// `milkie serve` can reuse it without examples depending on examples. This file
// re-exports it to keep agent-docs-qa's existing imports working.
export { BroadcastingEventStore } from '../../../src/trace/BroadcastingEventStore.js'
