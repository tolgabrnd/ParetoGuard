# Provider Compatibility Audit

**Date verified:** 2026-09-22 (Phase B.5, before Phase C). Verified against current
official documentation via web research — no live/paid API calls were made. This
document should be re-verified periodically; provider APIs change without notice
(see docs/LIMITATIONS.md).

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

**API surface used:** `generateContent` — `POST /v1beta/models/{model}:generateContent`
(kept; **not** migrated to the Interactions API)

**The question this audit had to answer:** Google's current docs literally title
the generateContent page "Gemini Generate Content API (Legacy)" and promote a
newer Interactions API (GA June 2026) as the default for new projects, especially
agentic ones. This is a real, confirmed deprecation-labeling — not a
misunderstanding. So: should ParetoGuard migrate?

**Decision: (c) — keep `generateContent`, with justification, revisit later.**

**Why not migrate now:**
- `generateContent` has **no announced sunset date** and Google states it "will
  continue to receive new mainline Gemini models for the foreseeable future."
  Only *frontier* agent-specific capabilities are expected to land Interactions-only
  going forward.
- `generateContent`'s schema was verified **consistently across three independent
  fetches** of official docs (contents/candidates/parts/usageMetadata field names
  matched every time) — high confidence.
- The Interactions API's schema, by contrast, was **inconsistent across sources**
  at verification time: one official page described response items under
  `execution_steps`, another under `steps`, and the exact request-array typing
  for stateless mode (`input: [...]`, item `type` values) was only available from
  one source with no independent corroboration. Given this project's explicit
  "never fabricate results/capabilities" rule and the instruction not to make live
  calls to verify empirically, implementing against a schema with unresolved
  naming conflicts would mean shipping a guess presented as verified fact — a
  worse outcome than staying on a well-documented "legacy" API.
- **The trade-off the task asked about** — "would Interactions prevent the
  normalized provider abstraction from working correctly?" — turned out to be
  *no*: Interactions supports a genuine stateless mode (`store: false`) that
  accepts a full multi-turn `input` array in one request, matching ParetoGuard's
  per-request abstraction just as well as `generateContent` does. So the decision
  to stay on `generateContent` is purely about verification confidence, not an
  architectural incompatibility with Interactions.

**Verified unchanged (no fix needed):**
- Auth header `x-goog-api-key: <key>` — confirmed current/recommended over the
  legacy `?key=` query param (and this adapter already avoided the query-param
  form specifically to keep the key out of any logged URL)
- Endpoint shape, `contents[]`/`systemInstruction`/`generationConfig`/`tools[].functionDeclarations`
  request fields, `candidates[].content.parts[]`/`finishReason` response fields,
  `usageMetadata.promptTokenCount`/`candidatesTokenCount`/`cachedContentTokenCount`

**Operational risk found (not an adapter bug):** Google is retiring "standard" API
keys in favor of identity-bound "auth" keys — unrestricted standard keys were
rejected starting 2026-06-19, and **all** standard keys starting September 2026
(i.e., now, at the time of this audit). This doesn't change any request/response
schema or header name — the adapter just passes through whatever key string is in
`GEMINI_API_KEY` — but a user hitting a 401/403 today may be running an old
"standard" key rather than hitting an adapter bug. The adapter's existing
generic 4xx → `PROVIDER_FAILURE(retryable=False)` classification already handles
this reasonably (a non-retryable failure is the correct behavior either way); no
code change was needed, but it's worth knowing when debugging.

**Revisit trigger:** once the Interactions API's response schema can be confirmed
from a single authoritative source without contradiction (or via an opt-in live
smoke test — never in the default test suite), re-run this trade-off analysis.

**Sources:**
- [Interactions API overview](https://ai.google.dev/gemini-api/docs/interactions-overview)
- [generateContent reference](https://ai.google.dev/api/generate-content)
- [Using Gemini API keys (generateContent/Legacy)](https://ai.google.dev/gemini-api/docs/generate-content/api-key)
- Standard-key retirement timeline, cross-checked across DoiT and Cybernews
  coverage of Google's own announcement (both cite the same June 19 / September
  2026 dates)

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
