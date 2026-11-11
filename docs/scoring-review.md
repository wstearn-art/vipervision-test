# Scoring review — findings only

The six-point scorer is no longer present. `score_tools` is a compatibility wrapper around plan evaluation; `comfort` and `job` arguments do not contribute numeric points. This implementation leaves `recommendation_engine.py` and both catalogs unchanged.

## Retain

- Exact plan capability coverage, documented evidence status, explicit excluded capabilities, and separate unknown features.
- Strong Fit only with confirmed cost and all identified essential capabilities covered; Partial Fit with confirmed cost, meaningful coverage, and explicitly documented exclusions for missing needs; Research Lead for uncertain costs/evidence or unsupported missing capabilities.
- Plan-level cost assembly, known-cost lower bounds, over-budget exclusion, seat counts distinct from employees, usage and overage checks, add-ons/base subscriptions, and billing restrictions.
- Fit-class ordering before coverage/preference/cost ordering, and a shared evaluation for recommendation cards, comparisons and ROI candidates.

## Weaknesses and recommendations requiring approval

1. **Unstated locations default to one.** Location-priced plans and location limits can appear confirmed without explicit location count. Treat the count as unknown when it materially affects eligibility or cost.
2. **Mandatory non-capability restrictions are represented as unresolved text.** This safely blocks a confirmed fit, but conflates missing customer clarification with missing vendor evidence. Introduce typed restrictions and distinguish those states if the scoring model is revised.
3. **Evidence status is trusted and does not enforce freshness or source completeness.** Verification dates and URLs are displayed but do not gate classification. Add catalog validation and an agreed freshness policy rather than silently changing fit labels.
4. **Per-tool evaluation is not a combined purchasing recommendation.** Each tool is compared with the full budget independently. Several recommended tools can exceed the budget if purchased together. Define bundle evaluation separately before presenting a combined solution as affordable.
5. **Industry and comfort do not affect eligibility.** This avoids category-based false exclusions, but beginner suitability remains advisory and industry suitability may be unverified. Decide whether either should become a documented requirement; do not restore arbitrary points.
6. **Optional-only requests have no essential capability anchor.** The engine can produce Research Lead classifications without meaningful essential coverage; display filtering can then leave no recommendations. Decide the desired optional-only behavior explicitly.
7. **Capability vocabulary is incomplete.** Several catalog capability IDs (e.g. generic drafting, estimating workflow and review monitoring) lack corresponding engine text patterns. Current structured interpretation intentionally uses the existing recognized vocabulary. Centralize an approved capability registry as a separate catalog/model change.
8. **The evaluation cache uses profile revision alone.** A catalog change during a persistent session can leave stale evaluations. Add a catalog fingerprint to cache identity in a separate evaluation-lifecycle change; this branch only invalidates on profile edits and migration.
9. **ROI savings assumptions persist across task changes.** Costs refresh, while hours, hourly value and estimated reduction retain existing widget state. Preserving them avoids an unapproved UI change, but a future task-specific scenario design would reduce confusion.
10. **Catalog evidence quality remains a research responsibility.** Tests establish consistency with supplied records, not that vendor claims or prices remain current. The engine's evidence-status whitelist is not independent verification.

The safer next step is to evaluate these changes individually with explicit policy decisions and dedicated tests. None is silently implemented in this architecture branch.
