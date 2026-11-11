# Structured customer profiles

Baseline: `wstearn-art/vipervision-test` at `8e170f9f0c44e8fa34566bc7c6d7ea717b1cc455`.
Review branch: `codex/structured-customer-profile`. No main-branch update or deployment.

## Change

Luna now proposes a JSON action envelope against a profile revision. Python validates the complete envelope, applies it to a copy, and publishes it only after every action passes. Requirements have server-assigned IDs, descriptions, capability IDs, and essential/optional status. Corrections target IDs; conversation history remains separate from active needs.

`customer_profile.py` owns the contract, validation, migration, and compatibility projection. `vee_chat.py` owns the model transport and Streamlit integration. Existing engine input fields are derived from requirements; they are not independently editable state. Unknown essential workflows remain unresolved; unknown optional workflows do not block essential coverage. Old unresolved fragments migrate with IDs for explicit reconciliation; migration does not silently discard them.

The updater increments revision once per changed turn, preserves the shared answers dictionary identity, and invalidates evaluation, research leads, inline replies, and ROI choices. Budget controls submit structured actions directly, without a model or natural-language parser. A missing, malformed, stale, conflicting, or ambiguous interpretation preserves the profile. Existing Luna-to-mini transport fallback is retained; invalid JSON after a successful transport is rejected rather than reinterpreted by regex.

## Consolidation

Removed `_conversation_control`, `_requirement_identity`, `_profile_changed`, `_number_of_people`, `_early_employee_count`, `_free_preference`, `_early_comfort`, `_explicit_business`, `_explicit_challenge`, `_explicit_goal`, `_monthly_budget`, `_budget_is_ambiguous`, `_budget_update_intent`, `_budget_label`, `_business_type`, `_comfort`, `_is_greeting`, `_optional_name`, `_purchasing_details`, `_hard_requirements`, and `_needs_goal_clarification` from the conversation layer. Replaced the large `_update_profile` parser with one interpretation request and one action application path. Legacy category callback arguments remain accepted for app compatibility but are no longer invoked for profile interpretation. The engine's text compatibility functions remain unchanged for legacy engine callers and one-time migration.

## Regression evidence

Tests were committed before production changes (`12d5007`). On unchanged production code, the original landscaping test failed with unresolved `I need help with crew coordination` and `assigning jobs`; five additional structured-profile contracts failed because that architecture did not exist. All five initial engine tests passed.

The suite covers intake; confirmations and questions; mixed questions/instructions; additions, removal, replacement and correction; unknown and optional needs; repeated paraphrases; atomic rollback; malformed/stale/unavailable model output; invalid capabilities and values; revision and invalidation; migration; direct budget controls; fit classes; seat/add-on/usage totals; billing and headcount limits; real catalog coverage; shared evaluation and comparisons; and a seven-message Streamlit landscaping flow with a subsequent budget-button update.

Run with `python -B -m unittest discover -s tests -v` after installing `requirements.txt`. This workspace uses a local dependency directory: `$env:PYTHONPATH = '.test-deps'`.

The Streamlit test with actual social-content recommendation cards requires `st.link_button(key=...)`, already used by the baseline research-lead controls. On local Python 3.9 / Streamlit 1.50 it is explicitly skipped: an initial execution exposed that existing incompatibility. The seven-message landscaping Streamlit test runs without that control and passes. Shared evaluation and comparison tests independently exercise nonempty real-catalog results. No UI workaround is included in production code.

Final local result: **24 tests run, 23 passed, 1 explicitly skipped**. No remaining failures. Live-model semantic evaluation and the skipped full-card flow remain release checks.

## Preservation

`app.py`, `recommendation_engine.py`, `style.css`, `ai_tools.csv`, `tool_plans.csv`, and `requirements.txt` remain byte-identical to baseline. Permanent Git-blob checks enforce this. AST fingerprints enforce unchanged card, comparison, chat-style functions and unchanged Streamlit presentation/control calls in `render_vee_chat`. Navigation, ROI formulas, buttons, fonts, styles, images and assets are not modified. The branch tree inherits binary assets directly from baseline; the local text-only source snapshot uses the existing avatar fallback.

## Limits and release risks

- Deterministic fixtures validate action processing, not live Luna semantic accuracy. No live model credentials were used. Run a live conversational evaluation before deployment, including paraphrases, pronouns, confirmations and adversarial instructions.
- Python checks capability IDs and action structure, not whether a known capability semantically covers a customer's entire request. The prompt requires unsupported restrictions to remain separate unknown needs; a model can still mis-map them. Catalog eligibility/pricing remains deterministic, but an inaccurate interpretation can supply inaccurate inputs.
- New descriptions and semantic deduplication rely on the model. Exact duplicate additions are idempotent; paraphrases must update an existing ID. Python deliberately does not introduce another language parser.
- If both model transports fail, intake pauses with a clarification and preserves state. The former regex-only intake fallback was removed with the parser. Existing budget controls remain deterministic. This operational tradeoff should be accepted before deployment.
- Migration retains old unknown fragments rather than guessing which ones to discard. A subsequent explicit model reconciliation can refine/remove them by ID. Existing sessions remain version 4 and migrate without resetting unrelated answers.
- The fixed intake stages and existing acknowledgment/product-answer mechanisms are retained. Mixed questions plus successful edits receive the existing update acknowledgment; the same-turn question is not guaranteed a separate answer.
- The baseline Streamlit research-link compatibility issue remains. Verify the deployment runtime before release; changing those controls requires the requested UI approval.
- No scoring-model changes, catalog research updates, UI changes, merge, or deployment are included.
