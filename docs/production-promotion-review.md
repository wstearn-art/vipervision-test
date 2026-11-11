# Reliability production promotion — merge blocked

Source: `wstearn-art/vipervision-development`, branch `codex/structured-customer-profile`, verified commit `8120fb740977ec071fc451bdfdcac3e01b15a5d1`.
Production base: `wstearn-art/vipervision-test/main`, `e55a16121d20707b58a3a2cf9a5bbeaa959f6bc4`.
Release branch: `release/reliability-fixes`.

## Scope and preservation

The application changes require `vee_chat.py` and `customer_profile.py`. Import inspection confirms the current production `app.py` and `recommendation_engine.py` already provide their dependencies; no additional runtime modules, catalog changes, requirements changes or UI changes are required.

The release also includes all six source regression test files, the architecture implementation report, the reliability report, and this promotion review. The older reports describe development-stage work and authorization at that time; this document records the current promotion decision. The development `.gitignore`, scoring review, local environments and local test logs are not transferred.

Production has two newer formatting-only commits: `20dfd6b` removes a comment's trailing period in `vee_chat.py`, and `e55a161` uses a single-quoted docstring literal in `customer_profile.py`. Three-way merging the private changes against their common module baseline succeeds without conflicts and retains both production edits. AST comparison confirms the resulting application modules have identical executable structure to the verified private version. Every other existing production file, including configuration, catalogs, styling and binary assets, is retained from the current production tree.

## Test results and merge blocker

1. Python 3.9.13 / Streamlit 1.50: full release suite, **42 run, 41 passed, 1 skipped, 0 failures/errors**. The existing full-card test is skipped because that Streamlit version lacks `link_button(key=...)` support.
2. Python 3.14.6 / Streamlit 1.67.0: deterministic suite, **38 run, 5 failing assertions across 2 preservation test methods, 0 errors**. Four assertions fail in the style/card/action fingerprint method and one fails in the chat-control fingerprint method. The other 36 test methods pass, including the byte-level engine/catalog/app/CSS preservation test.
3. A full Python 3.14 run stalled in its first Streamlit AppTest fixture and was interrupted. The four Streamlit test methods therefore do not have a completed modern-runtime result. An earlier diagnostic run before dependency installation completed produced import errors and is not treated as a valid completed environment test.

The fingerprint failures are reproducibly Python-version-dependent: `ast.dump` on 3.9 emits empty fields such as `arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])`; 3.14 emits `arguments()`. The checked-in expected hashes were generated on 3.9. Identical source therefore fails those assertions on 3.14. The assertions were not changed, skipped or weakened for this release. This does not demonstrate a styling modification, but it means the release does not satisfy the owner's all-tests-pass merge condition.

**Do not merge this release yet.** Make the preservation tests version-independent and diagnose the modern AppTest stall in private development, then rerun applicable validation before resuming promotion. No application correction or test-only portability fix has been silently included here.

## Remaining conversational risk

Independent live customer-flow QA was blocked by staging access restrictions. Mocked automated tests do not establish that all conversational defects are resolved. Semantic review still depends on model accuracy and adds latency and token use. A successful production startup would not resolve that remaining QA gap.

## Deployment and branch handling

No merge, deployment, branch deletion, or staging configuration change is performed by this promotion attempt. Public main remains at the recorded production base. Private branches and the existing staging deployment are unchanged. Production deployment verification of this release remains pending because the merge is blocked.
