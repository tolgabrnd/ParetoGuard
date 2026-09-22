# Provider Compatibility Audit

**Date verified:** 2026-09-22 (Phase B.5, before Phase C; Gemini section re-verified
and corrected the same day — see "Gemini" below). Verified against current official
documentation via web research — no live/paid API calls were made. This document
should be re-verified periodically; provider APIs change without notice (see
docs/LIMITATIONS.md).

## Why this audit happened

The Phase B adapters (`src/paretoguard/providers/{openai,anthropic,gemini}.py`) were
originally implemented from model knowledge rather than verified current
documentation. This audit checked each provider's current official docs and fixed
anything that had actually drifted.

## OpenAI

**API surface used:** Responses API — `POST /v1/responses`
(was: Chat Completions — `POST /v1/chat/completions`)

**Why:** Chat Completions is **not deprecated** and still works — only the older
Assistants API is being sunset (2026-08-26), and Responses is its replacement, not
Chat Completions'. However, OpenAI's own migration guide states Responses is
"recommended for all new projects," specifically citing agentic workflows, better
reasoning-model performance, and improved prompt-cache utilization (40–80% better
than Chat Completions per OpenAI's internal tests). ParetoGuard is squarely an
agentic/tool-use evaluation platform, so this recommendation is directly relevant,
and the schema was verified consistently across multiple independent official
sources (developers.openai.com migration guide, API reference, and a documented
structured-outputs example) — enough confidence to migrate for real rather than
leave a stale adapter in place.

**What changed in the adapter:**
- Endpoint: `/v1/chat/completions` → `/v1/responses`
- Request: `messages` array → `input` array of items; system prompt moved from an
  in-array `role: "system"` message to a top-level `instructions` string
- `max_tokens` → `max_output_tokens`
- Tool schema: no longer wrapped in `{"type": "function", "function": {...}}`; now
  flat — `{"type": "function", "name", "description", "parameters"}`
- Structured output: `response_format` → `text.format` (`{"type": "json_schema",
  "name", "schema", "strict"}`)
- Response: `choices[0].message.content` → `output` array of typed items
  (`type: "message"` with `content[].type == "output_text"`, or
  `type: "function_call"` with `call_id`/`name`/`arguments`); there is **no**
  top-level `output_text` string field in the raw JSON — that's an SDK-only
  convenience property, so the adapter builds its own equivalent by scanning
  `output`
- `usage.prompt_tokens`/`completion_tokens` → `usage.input_tokens`/`output_tokens`;
  cached tokens moved to `usage.input_tokens_details.cached_tokens`
- New `status` field (`completed`/`failed`/`incomplete`/...) replaces per-choice
  `finish_reason` as the primary success/failure signal
- Auth header unchanged: `Authorization: Bearer <key>`

**Model compatibility:** the adapter has no hard-coded model names — `model` is
config-driven (`configs/models.example.yaml`), so newer OpenAI models work without
adapter changes as long as they're served through the Responses API.

**Known risks:** OpenAI's exact `incomplete_details.reason` enum values weren't
independently verifiable without a live call, so the adapter maps *any*
`status: "incomplete"` to `FinishReason.LENGTH` rather than trying to distinguish
sub-reasons — a defensible simplification, not a guess dressed up as verified fact.

**Sources:**
- [Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses)
- [Responses API reference](https://developers.openai.com/api/reference/python/resources/responses)
- Structured-outputs `text.format` shape, cross-checked via a documented `responses.create` example (medium.com/@alexanderekb, corroborated by the official migration guide's "response_format deprecated, use text" note)

## Anthropic

**API surface used:** Messages API — `POST /v1/messages` (unchanged)

**Why:** this is Anthropic's current, only public API for text generation. No
newer replacement exists.

**What changed in the adapter:** authentication only —
`x-api-key: <key>` → `Authorization: Bearer <key>`. Per platform.claude.com's
authentication docs (fetched directly, not just search snippets): "Send it as
`Authorization: Bearer <key>` on direct HTTP requests" is the current recommended
method; `x-api-key` is explicitly documented as "legacy" but still supported.
Both work identically today, so this is a low-risk change that follows current
guidance rather than a compatibility break.

**Verified unchanged (no fix needed):**
- Base URL `https://api.anthropic.com`, endpoint `/v1/messages`
- `anthropic-version: 2023-06-01` header
- Request fields: `model`, `max_tokens`, `system`, `messages`, `tools`,
  `temperature`
- Tool schema: flat `{"name", "description", "input_schema"}` — **no** `"type"`
  field required for custom/client tools (confirmed against a live example on
  platform.claude.com's tool-use overview page, which contradicted an earlier,
  less authoritative search-summary claim that `"type": "custom"` was required —
  this is why the adapter fetches primary doc pages rather than trusting a single
  secondary source)
- Response fields: `content` blocks (`text`, `tool_use`), `stop_reason` values
  (`end_turn`, `stop_sequence`, `max_tokens`, `tool_use`), `usage.input_tokens` /
  `output_tokens` / `cache_read_input_tokens`

**Known risks:** `temperature` is documented as deprecated for models released
after Claude Opus 4.6 (fixed at 1.0 for backwards compatibility) — not an adapter
bug since the adapter only sends `temperature` when the caller sets one, but worth
knowing when configuring newer models.

**Sources:**
- [Authentication](https://platform.claude.com/docs/en/manage-claude/authentication)
- [Messages API reference](https://platform.claude.com/docs/en/api/messages)
- [Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)

## Gemini

**API surface used:** Interactions API — `POST /v1beta/interactions`, stateless
mode only (`"store": false`, no `previous_interaction_id`)
(was: `generateContent` — see "Revision history" below)

**Why this changed from the initial Phase B.5 decision:** the first pass of this
audit kept `generateContent`, citing schema inconsistency across sources for the
Interactions API as the blocking concern (see git history / revision note below
for that original reasoning). A follow-up re-verification pass resolved that
inconsistency: `ai.google.dev/api/interactions-api`'s field-schema table (fetched
directly, not a secondary summary) explicitly documents `input` as accepting
`Content | array(Content) | array(Step) | string`, and a dedicated function-calling
guide gave a complete, internally consistent round-trip example (tool declaration,
`function_call` step, `function_result` step). With that resolved, the balance of
evidence favored migrating: `generateContent` is explicitly labeled "(Legacy)" in
its own page title, Interactions has been GA since June 2026 and is documented as
the default for new projects, and its stateless mode is a first-class, fully
documented option — not a workaround.

**Stateless-mode rationale:** Interactions' *recommended* mode stores conversation
history server-side, referenced via `previous_interaction_id` on later calls. That
doesn't fit `paretoguard.providers.base.Provider`, which is a stateless
`InferenceRequest` in / `InferenceResponse` out contract with no adapter-held
session — adding session semantics would mean leaking a Gemini-specific concept
into the generic interface, which the task explicitly ruled out. Google's own
function-calling guide confirms `store: false` is not a degraded fallback: a
single stateless request can carry a full `function_call`/`function_result`
history in `input`, exactly like the OpenAI and Anthropic adapters already do by
resending full message history every call. So every Gemini call from this adapter
sends `"store": false` and never references `previous_interaction_id`.

**What changed in the adapter:**
- Endpoint: `/v1beta/models/{model}:generateContent` → `/v1beta/interactions`
  (model moves from the URL path into the `model` request field)
- Request: `contents[]` (role-tagged) → `input[]` of typed items. A bare
  `{"type": "text", "text": ...}` item is implicitly the caller's turn (no role
  field exists on `Content` items); a prior *assistant* turn must instead be
  replayed as a `{"type": "model_output", "content": [...]}` **Step**, or the API
  would read it as another user turn. Tool results become
  `{"type": "function_result", "call_id", "name", "result"}` steps.
- `systemInstruction: {"parts": [...]}` → `system_instruction` as a plain string
- `generationConfig.maxOutputTokens`/`temperature` → `generation_config.max_output_tokens`/`temperature`
- `tools[].functionDeclarations[]` → flat `tools: [{"type": "function", "name",
  "description", "parameters"}]` (same flat shape now used by the OpenAI adapter)
- Response: `candidates[].content.parts[]`/`finishReason` → `steps[]` of typed
  items (`model_output`, `function_call`, `function_result`, `thought`, ...) plus
  a top-level `status` (`completed`/`incomplete`/`failed`/...)
- `usageMetadata.promptTokenCount`/`candidatesTokenCount`/`cachedContentTokenCount`
  → `usage.total_input_tokens`/`total_output_tokens`/`total_cached_tokens`
- Auth header **unchanged**: `x-goog-api-key: <key>` (Interactions uses the same
  header as `generateContent` did — this is not affected by the standard-vs-auth
  key transition below, only the *value* is)

**Normalized fields supported:** `output_text` (joined `model_output` step text),
`tool_calls` (from `function_call` steps), `finish_reason` (from `status`, with
`TOOL_CALLS` taking priority when tool calls are present), `token_usage` (from
`usage`), `error` (from in-band `status: "failed"` + `errors[]`, or from
transport/HTTP-level failures via the shared `providers/http.py` classifier),
`latency` (measured locally, as with every adapter).

**Left in raw metadata, not normalized:** `thought` steps (extended-reasoning
traces) have no equivalent in `InferenceResponse` — no normalized "reasoning"
field exists, and adding one solely for Gemini would mean redesigning a core type
around one provider's concept, which the task explicitly ruled out. Instead, the
adapter sets `raw_provider_metadata = {"id": ..., "steps": ...}`, preserving the
full, unprocessed `steps` array (thought steps included) for any caller that
wants them, without losing that information silently.

**Known residual uncertainty:** the `function_result` step's exact field set
showed a minor inconsistency across official doc pages during verification — one
example omitted `call_id`, a more detailed one (the dedicated function-calling
guide) included it, correlating to the originating `function_call`'s `id`. This
adapter sends `call_id`, matching the more detailed source; documented in the
adapter's own docstring as the first thing to check if Gemini rejects or ignores
that field. No live call was made to resolve this (per the task's constraint), so
this is flagged rather than guessed past.

**Verified, not changed by this migration:** the September-2026 standard-vs-auth
API key transition (unrestricted standard keys rejected since 2026-06-19, all
standard keys rejected starting September 2026) is unaffected by which endpoint
is used — it's about the credential *value*, not the request shape or header
name. Still an operational risk for whoever configures `GEMINI_API_KEY`, not
something adapter code can detect or fix; the shared HTTP error classifier's
generic 4xx → `PROVIDER_FAILURE(retryable=False)` already handles an auth
rejection reasonably.

**Sources:**
- [Interactions API reference](https://ai.google.dev/api/interactions-api) — authoritative field-schema table resolving the `Content`-vs-`Step` `input` typing
- [Interactions API overview](https://ai.google.dev/gemini-api/docs/interactions-overview)
- [Function calling with the Gemini API](https://ai.google.dev/gemini-api/docs/interactions/function-calling) — full stateless tool-call round-trip example
- [Interactions API quickstart](https://ai.google.dev/gemini-api/docs/interactions/quickstart) — endpoint, `x-goog-api-key` header, response shape
- [generateContent reference (now legacy)](https://ai.google.dev/api/generate-content) — confirms no announced sunset date for the old API
- Standard-key retirement timeline, cross-checked across DoiT and Cybernews
  coverage of Google's own announcement (both cite the same June 19 / September
  2026 dates)

### Revision history

- **2026-09-22, first pass:** decided to keep `generateContent`, citing
  unresolved `execution_steps`-vs-`steps` naming inconsistency across two
  sources as insufficient confidence to migrate.
- **2026-09-22, same day, corrected:** re-verified directly against the
  Interactions API's authoritative field-schema reference and a dedicated
  function-calling guide, which resolved the inconsistency (`steps` is correct;
  `execution_steps` was prose in an overview page, not a field name) and
  provided a complete, internally consistent request/response example. Migrated
  to Interactions on that basis. This revision history is kept rather than
  deleted so the reasoning trail — including what changed and why — stays
  auditable.

## `pyproject.toml` changes

Removed the `openai`, `anthropic`, and `gemini` optional-dependency extras: none
of the three adapters import the official SDKs (all three call REST endpoints
directly over `httpx`, specifically so they're testable with
`httpx.MockTransport` without SDK-specific mocking — see each adapter's
docstring). Declaring unused extras was misleading. `torch` remains, for the
planned optional PyTorch router baseline.

## What this audit did not change

- Live-provider tests remain opt-in only (`@pytest.mark.live`, none exist yet,
  none run in CI) — this audit added zero live calls.
- No model IDs or pricing were added or changed; `configs/*.example.yaml` still
  intentionally leave real-provider entries as templates (see Phase A rationale
  in `docs/BUILD_PLAN.md`).
