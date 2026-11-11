# Reliability investigation and implementation

Repository: `wstearn-art/vipervision-development` (private).
Baseline: `codex/structured-customer-profile`, `fa6194ab43aa7d64839cfe2e19750fccc2ae2fbf`.
Working branch: `codex/reliability-improvements`.

The initial workspace was an unversioned source snapshot. Local HTTPS Git authentication was unavailable, but the existing GitHub connector had private read/write access. The connector restored the baseline's files, and Git verified the exact tree and commit objects in a shallow local checkout. The checkout's only remote is the private development repository. Remote branch operations use the connector. No public repository operation, merge or deployment is part of this assignment.

## Root causes and corrections

### Profile and intake interpretation

Confirmed code defect: `apply_proposal` accepted structurally valid requirements with descriptions such as `Wait`, retained acquisition when the proposal omitted a requested removal, and allowed paraphrases with identical mapped capabilities. Structural validity alone does not establish agreement with customer intent. The interpreter could omit an ordinary scalar answer or put personal details in the business field; Python could not detect the semantic mistake. These are model-dependent interpretation failures with deterministic acceptance paths, not evidence that every live response was wrong.

The existing action schema, stable IDs and atomic updater are retained. An additional bounded semantic review receives the original profile, latest message and recent context, candidate actions, candidate profile, and any validation error. It must return the same action contract against the original revision, correcting missed edits, wrong fields, filler and unanswered questions or requesting clarification. Only the final structurally validated proposal reaches the authoritative updater. A failed review preserves all saved state. No independent regex parser, new model, new state store, or dependency was introduced. Python also rejects duplicate mapped-capability sets or normalized duplicate descriptions in the final state; the reviewer can reconcile a rejected duplicate addition back to an existing ID.

The review reuses the existing Luna/fallback transport. It is a model check, not a mathematical guarantee of correct interpretation. Mocked tests demonstrate corrected/rejected proposal handling, not live-model reliability. Correlated mistakes remain possible.

Confirmed intake defect: stage advancement looked only after the current stage, allowing missing earlier information to be skipped. Advancement now selects the first genuinely unanswered field from the authoritative profile. The contract explicitly separates business, name, headcount and comfort, accepts noncatalog industries, and requires common budget/headcount answers to populate their own fields. Acknowledgment prompts may acknowledge only saved details.

### Mixed questions and edits

Confirmed routing defect: the successful-edit path constructed an update acknowledgment and bypassed `_followup`, discarding questions. The action envelope now carries validated questions separately from actions, with pricing/comparison/general kinds. The renderer answers those questions against the refreshed evaluation and then combines the answer with the actual update result. This also works during intake and alongside a valid clarification that leaves state unchanged. Direct budget controls still use the deterministic updater without either model call.

### Plan explanations and ROI

Confirmed grounding weakness: product-wide `price_note` text and plan-specific assessments were both supplied to generated follow-ups, allowing an advertised free plan to be substituted for the assessed paid plan. Named pricing questions now get deterministic answers from the current engine decisions, including the assessed plan, confirmed total or uncertainty/ineligibility, and the engine's complete-cost reason. Generated follow-ups receive plan assessments without product-wide price notes. Comparison text names the assessed plan and distinguishes published notes about other plans. Catalog-membership responses can still quote published notes as catalog information, not as proof that that plan meets the saved request.

The Comp-versus-Pro ROI arithmetic is not a scoring defect: keeping the same assumed savings while reducing cost necessarily increases hypothetical net savings. The missing qualification was coverage. ROI candidates now carry the current plan's fit label, covered and missing capabilities, and explicit instructions to exclude unsupported work and revise scenario assumptions when coverage changes. This is delivered through the existing caption field. The formulas, input values, controls, layout and pricing decisions are unchanged; the application does not infer a measured savings rate from fit classification.

Confirmed summary weakness: the priority summary echoed free-form requirement descriptions, which could embed an old budget. Mapped requirements now project their typed capability labels into summaries; budget is represented by the separate current budget field. Unmapped needs retain their descriptions for clarification. Semantic review is instructed to keep scalar metadata out of workflow descriptions.

## Removed or consolidated logic

- Removed the unused `ambiguous_budget` output and unreachable response branches left over from the old parser.
- Removed the unused `budget_edit` routing output and current-stage argument from `_next_unanswered`.
- Replaced forward-only stage advancement with one profile-based readiness decision.
- Replaced named-product generated fallback price claims with the same deterministic question-answer path.
- Removed product-wide price notes from generated follow-up grounding.
- Retained one authoritative updater; the semantic reviewer does not publish intermediate state.

Production diff relative to baseline: **137 lines added, 34 removed; net +103** across `customer_profile.py` and `vee_chat.py`. The extra production code is the review boundary, validated question contract, shared question routing and plan/coverage explanations. This is not a return to keyword patches.

## Verification

Baseline suite: **24 run, 23 passed, 1 skipped**.

Nine initial reconstructed regression methods, executed before production edits, reported **6 failures and 4 errors** (failure counts include subtests). They reproduced acceptance of filler, omitted removals and scalar fields, duplicate capabilities, forward-only stage progression and absent mixed-question/ROI mechanisms. Exact independent live transcripts were unavailable; all new conversation scenarios are explicitly reconstructed.

Final full suite: **42 run, 41 passed, 1 skipped, 0 failures, 0 errors**.
Command: `python -B -m unittest discover -s tests -q`, with the workspace dependency directory on `PYTHONPATH`.

New coverage includes controlled bad proposals followed by corrected reviews; failed review rollback; duplicate reconciliation; gift-shop budget and combined beginner/sole-employee intake; preservation of business details; mixed-message question extraction and actual Streamlit routing; complete Jobber plan-cost uncertainty; Buffer Essentials versus advertised Free; partial-fit ROI caveats; stale budget summaries; landscaping removal, dispatch refinement, missed calls and reintroduction; Cactus Starter with inventory forecasting added/removed; and 7shifts Pro changing to Comp while both essential needs remain saved. Shared evaluations are checked against the updated profile revision.

The existing Streamlit test fixture now reads the profile JSON before review metadata. Its behavior assertions are retained. The old exact comparison-text fingerprint was replaced by checks for the assessed plan and qualification of other-plan notes, because that text is intentionally corrected. Existing style/card/action fingerprints and all Streamlit presentation/control-call fingerprints still pass. `app.py`, `recommendation_engine.py`, requirements, catalogs and CSS remain byte-identical to baseline. All images and other assets are unchanged.

The one skipped test is the existing full recommendation-card flow on Python 3.9 / Streamlit 1.50, whose `link_button` does not accept the baseline `key` argument. No UI workaround or dependency upgrade is included. The existing landscaping Streamlit flow and two new routing/intake Streamlit tests pass. New isolated routing tests intentionally use no products; real catalog/pricing behavior is exercised separately.

## Remaining limits and release considerations

- No live model calls or lengthy live-site tests were performed. Independent live evaluation remains required, especially for ambiguous references, missing edits, confirmations, filler, and compound questions.
- Every interpreted turn now makes two bounded model requests before publication. This increases latency, token use and exposure to timeouts; the existing transport can try Luna and its unchanged fallback for each request. A review failure intentionally pauses intake rather than saving the unchecked candidate.
- The same model can repeat an interpretation mistake. Python guarantees schema/ID/atomicity/duplicate rules, not semantic truth. Unknown workflows still rely on model descriptions and review; there is no phrase blacklist pretending to solve natural language.
- Repeated mapped capability sets are treated as duplicate needs. Distinct subworkflows under one broad capability should be consolidated under one ID, with separately unverified restrictions preserved as unknown requirements.
- ROI remains a hypothetical scenario calculator. Missing coverage is now explained, but savings assumptions are not automatically reduced for Partial Fit and may persist across changes. No capability-to-savings model was invented.
- Valid legacy envelopes without `questions` remain accepted for direct controls and fixtures. Natural-language interpretation prompts require questions; if both interpretation and review omit one, it can still go unanswered.
- Engine decisions and vendor evidence were not changed or re-researched. Catalog freshness and the previously documented runtime compatibility issue remain separate concerns.
- No merge or deployment is authorized by this implementation report.

## Files changed

`customer_profile.py`, `vee_chat.py`, `tests/test_preservation.py`, `tests/test_streamlit_flow.py`, new `tests/test_reliability.py`, new `tests/test_reliability_streamlit.py`, and this report.
