# Lineage Taxonomy

Controlled vocabulary for `object.created` / `relation.created` events (#37 / #38).
Without a closed vocabulary every example invents its own `type` strings and
cross-run lineage queries become meaningless. This document is the contract that
#37 / #38 implement against, and that #40 (and future lineage consumers) query.

> **Status:** v1, scoped to what citation/provenance needs (`passage` + `cites`).
> Other types are declared here so the framework is whole, but only the ones
> marked **v1** must ship in #37/#38.

## Why this is a closed vocabulary

A `type` field that anyone fills freely (`passage` vs `chunk` vs `text`) makes the
event log unqueryable: "which sources did this run cite?" has no stable predicate
to match. The vocabulary below is the allowed set; producers MUST use a value from
it. Adding a value is a deliberate amendment to this doc, not an ad-hoc choice at
the call site.

## Object types

An **object** is a content-addressable artifact an agent read or produced. The
`object.created` event mints an `objectId` handle for it (see ARCHITECTURE.md
§Concept Model → Object). `type` MUST be one of:

| type | v1? | meaning | example (三国) | required `meta` |
|---|---|---|---|---|
| `passage` | **v1** | a contiguous slice of a document, with a line range | `read_file(chapter-49, 4-5)` → those two lines | `file`, `lineStart`, `lineEnd` |
| `file` | — | a whole file as one artifact | the entire `chapter-49.txt` | `file` |
| `claim` | — | a statement an agent generated | an answer sentence「曹操败于火攻」 | `text` |
| `artifact-blob` | — | an opaque produced blob (image, json, …) | a generated report | `mime?`, `bytes?` |

**Note on `passage` meta — the whole point of the model:** `lineStart`/`lineEnd`
come from the **real `read_file` call arguments**, not from text the agent wrote.
That is why representational drift (繁简 / wrong line number / merged lines) cannot
occur: the citation's origin is the tool-call record, not the agent's prose.

## Relation types

A **relation** is a typed, directed edge between two objects. The
`relation.created` event records it (see ARCHITECTURE.md §Concept Model → Relation).
`type` MUST be one of:

| type | v1? | meaning (`from` → `to`) | example |
|---|---|---|---|
| `cites` | **v1** | `from` references / is sourced from `to` | claim「曹操败于火攻」 **cites** passage(ch-49:4-5) |
| `derives_from` | — | `from` was computed/transformed from `to` | a summary `derives_from` a passage |
| `supersedes` | — | `from` replaces `to` | a corrected claim `supersedes` an earlier one |
| `equivalent_to` | — | `from` and `to` are the same artifact under different ids | a re-read passage `equivalent_to` a prior one |

## Field set

Shared object/relation fields (carried in the event payloads, see #37/#38):

- **object.created:** `objectId`, `type` (above), `producerEventId` (the real
  `tool.responded` / `llm.responded` / `agent.returned` that produced the content),
  `hash?` (aligns with #25 `outputHash` when alignable), `meta` (type-specific,
  per the tables above).
- **relation.created:** `relationId`, `type` (above), `fromObjectId`, `toObjectId`,
  `causedByEventId`, `meta?`.

**Extension points (forward-compatible with the Context Layer):** `meta` MAY carry
`tag`, `source-uri`, `version`. These are reserved names; v1 does not require them
but producers must not repurpose them.

**Extensible types (#113 P4).** `ObjectType` / `RelationType` are no longer hard
closed unions: the core kinds above stay the framework's controlled vocabulary,
but an application MAY introduce its own kind using a `namespace:kind` convention
— e.g. `code:function`, `db:row` (objects), `app:tested_by` (relations). The
namespace keeps cross-run queries meaningful: a consumer can group by core kind
*and* distinguish app kinds, instead of colliding in one flat space. Core kinds
carry no namespace; app kinds MUST. Apps that need a custom-typed object/relation
define their own producer tool (using `ctx.createObject` / `ctx.createRelation`);
the framework's built-in `cite` / `declare_relation` cover the core vocabulary.

> **Deferred (not built — avoids premature abstraction).** The taxonomy also
> anticipates splitting an object's type into a closed `role` (source / claim /
> derived) and an open `resourceKind` (passage / file / function / …), plus a
> `resourceKind → resolver` registration so a UI/verifier can fetch any cited
> resource's real bytes. With a single resource kind (`passage`) today, building
> that registry would be premature; it lands when a second resource kind exists.
> See #113 (P4 scope note).

## Emit hook locations (三国 example — indicative, not implemented by #39)

- **`object.created` (passage):** `examples/agent-docs-qa/tools/corpus-tools.ts`
  `read_file` / `grep` handlers → on `tool.responded`, call `ctx.createObject({ type:
  'passage', meta: { file, lineStart, lineEnd }, hash: outputHash })` and return the
  minted `objectId` in the tool result so the agent can later cite it. (#37)
- **`relation.created` (cites):** the agent declares a citation via a `cite(objectId)`
  tool, which calls `ctx.createRelation({ type: 'cites', from: <answer claim>, to:
  <passage objectId> })`. (#38 provides the API; the new sanguo-researcher issue is
  the real producer.)

## See also

- ARCHITECTURE.md §Concept Model → `Object`, `Relation` (one-line canonical defs).
- `docs/lineage-lifecycle.md` (取证→cite→落账→消费的完整时序图与 lazy-promote 状态机).
- `docs/design/40-lineage-citation-goal.md` (why this exists; the burn-down target).
- Issues: #39 (this doc), #37 (`object.created`), #38 (`relation.created`), #40
  (front-end consumes these events).
