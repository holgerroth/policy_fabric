# Local validation

## MLCommons repository migration — 2026-11-09

Base: `mlcommons/policy_fabric`, `authorities@31776f3df799047d001067271a1155e21bda1cc4`.
Environment: macOS ARM64, Python 3.12.9, NVFlare 2.9.0, NumPy 2.5.4.
Run: `runs/20261109-policy-fabric-migration-01/` (ignored local artifacts).
This was a **local CPU run**, not an H100 or PDO ledger run.

All four cases used `Recipe.execute(PolicyProdEnv(before_submit=...))`, real
NVFlare TLS authentication and signed jobs, and explicitly selected mock
capabilities/Guardians. Actual registration metadata carried `asset_did`, and
that metadata was asserted in the deployed workspaces.

| Case | Job ID | Outcome | Client executions |
| --- | --- | --- | --- |
| Approved | `613c107d-5985-4270-b9fd-e2da165e031a` | `FINISHED:COMPLETED` | 2 |
| Re-signed tampered ZIP | `0a1209f3-16bd-4132-963b-f4652a5ad6ba` | `FINISHED:FAILED_TO_RUN`; Guardian HTTP 422 | 0 |
| Missing capability | `3ca45764-6cae-49e1-8ab7-1f0c6cef00c0` | `FINISHED:CAN_NOT_SCHEDULE`; site authorization denial | 0 |
| Guardian unavailable | `28bded4e-14a3-4417-bd7d-f64dd0e6df54` | `FINISHED:FAILED_TO_RUN`; deployment denied | 0 |

The approved job's `Run.get_result()` output contained the exact asserted model
`[[2, 3], [4, 5]]`. `results.json` records `passed: true`.

`python -m pytest -q tools/dev/nvflare_poc/tests`: **58 passed**.
Black (120 columns), isort (Black profile), and Git whitespace checks passed.
The updated SVG was rendered to PNG and visually inspected.

The focused tests cover both explicit Guardian modes, the PDO top-level
capability envelope, rejection of mock decisions in PDO mode, measured-digest
binding, registration routing, native denial preservation and byte preservation.
The tests execute the current upstream `InferenceGuardianActionRunner` and
`InferenceOperation`, while stubbing external services/PDO validation. They also
cover exact digest/role forwarding, policy/readiness refusals without fallback,
invalid targets and approval of a frozen export without changing its ZIP bytes.
Those contract tests are **not cryptographic or live-policy validation**.

**Remaining validation:** real authority credentials and policy evaluation,
encrypted capability issuance/decryption, Guardian audience/trust checks, and
NVFlare execution in one PDO-backed flow. PDO and Docker were not available in
the local demo environment. The installed client bridge requires the separately
configured PDO/Django environment described in the README.

The existing secret scan remains enabled; its high-entropy warnings concern the
explicit mock capability strings. Historical results below describe the older
identifier/Guardian contracts and are retained as historical evidence only.

## Recipe submission hook — 2026-11-06

Environment: macOS ARM64, Python 3.12.9, NVFlare 2.9.0, NumPy 2.5.4.
Run: `runs/20261106-recipe-hook-01/` (ignored local artifacts).
This was a local CPU test, not an H100 test.

Every case was submitted through a standard `NumpyFedAvgRecipe` using
`recipe.execute(HookedProdEnv(before_submit=...))`. The callback prepared the
final temporary export before normal Flare API submission. Secure-mode
authentication, signed jobs, and the unchanged upstream mock Guardian remained
enabled.

The example class was subsequently renamed to `PolicyProdEnv` for clarity.
The recorded run's `submission` field retains the original `HookedProdEnv` name;
execution behavior is unchanged. All 12 submission-hook cases passed again
after the rename.

| Case | Job ID | Outcome | Client script executions |
| --- | --- | --- | --- |
| Approved | `b405146d-96ba-4481-8577-e93cf7dfcb97` | `FINISHED:COMPLETED` | 2 |
| Re-signed tampered payload | `e82d4114-2d3f-4b73-8a11-2e64cb17113d` | `FINISHED:FAILED_TO_RUN`; Guardian HTTP 422 | 0 |
| Missing capability | `4327a6f0-66fb-4bb5-8e5d-24d888974e12` | `FINISHED:CAN_NOT_SCHEDULE`; site authorization denial | 0 |
| Guardian unavailable | `ae75e51e-cc46-47e8-bbb9-11fd7c0dd572` | `FINISHED:FAILED_TO_RUN`; deployment blocked | 0 |

The approved run retrieved its result workspace through the inherited
`Run.get_result()` API. The harness loaded `server.npy` and asserted that the
aggregated model was exactly `[[2, 3], [4, 5]]`.

`python -m pytest -q tools/dev/nvflare_poc/tests`: **42 passed** (30 existing
integration checks plus 12 submission-hook cases). The new cases cover a real
Recipe export reaching the hook, final files reaching submission, failure before
connection, invalid callback returns, session/directory cleanup, no-hook behavior,
fresh exports on repeated execution, export-only behavior, inherited status
lookup, and capability issuance over the signed frozen bytes. Tests that bind
local HTTP sockets required execution outside the filesystem/network sandbox.

The normal NVFlare secret scanner was preserved and emitted warnings about the
mock base64 capability strings. Those are expected test artifacts, not trusted
credentials. Black (120 columns), isort (Black profile), and whitespace checks
were run on the changed Python files.

## Original explicit export/submission baseline — 2026-10-15

Environment: macOS ARM64, Python 3.12.9, NVFlare 2.9.0, NumPy 2.5.4.
Policy Fabric baseline: `fd66bb3c6526bdff16873aafa26e3299b59f74c0`.
Test run: `runs/20261015-04/` (ignored runtime artifacts, retained locally).
This was a local CPU test, **not an H100 test**.

## End-to-end results

| Case | Job ID | Outcome | Client script executions |
| --- | --- | --- | --- |
| Approved | `bbcda605-81a2-4077-a170-941b731be545` | `FINISHED:COMPLETED` | 2 |
| Re-signed tampered payload | `46adfee6-2dfd-4fd1-9f1c-bb3d57de1e1f` | `FINISHED:FAILED_TO_RUN`; both Guardians denied with HTTP 422 | 0 |
| Missing capability | `8d6ea9b9-c348-4056-9ffe-07bfe2d15678` | `FINISHED:CAN_NOT_SCHEDULE`; site B denied via job authorization | 0 |
| Guardian unavailable | `d43f860e-b226-4724-98cf-85b50c657146` | `FINISHED:FAILED_TO_RUN`; both clients denied before deployment | 0 |

The accepted job performed one FedAvg round, changing the initial NumPy model
`[[1, 2], [3, 4]]` to `[[2, 3], [4, 5]]`. The final array was read from
`models/server.npy` inside the saved server workspace ZIP and checked exactly.

Actual deployed runtime metadata contained both registration-advertised IDs:

```json
[
  {"name": "site_a", "site_config": {"policy_id": "did:pdo:policy-a"}},
  {"name": "site_b", "site_config": {"policy_id": "did:pdo:policy-b"}}
]
```

The tampered payload and its outer submission were signed with the provisioned
NVFlare submitter key **after** changing the payload, but the prior Guardian
capability was retained. Both clients therefore passed native signature checks
and were rejected specifically by the Guardian digest check. This distinguishes
policy approval from merely having permission to sign/submit a job.

NVFlare secure-mode authentication and its native signed-job requirement stayed
enabled. The demonstration used the unmodified upstream mock Guardian; no real
PDO credential, encryption, audience, lifetime, or replay checks were tested.

## Focused validation

`python -m pytest -q tools/dev/nvflare_poc/tests`: **30 passed**.

Coverage includes byte/file hash equivalence, actual upstream mock HTTP calls,
registration-based routing, missing/mismatched policies and application routes,
preserving existing denials, no unpacking after a failed Guardian check, response
schema/digest validation, HTTP failures, redirects, timeouts, unreachable
Guardians, and preserving frozen bytes through NVFlare's repacking path.

Black (120 columns), isort (Black profile), and `git diff --check` passed.

Earlier development runs remain under `runs/20261015-01` through `-03` for
diagnostics. Run `-04` is the validated result: it adds specific failure-reason
assertions so an unrelated native rejection cannot masquerade as a successful
Guardian hash-check test.
