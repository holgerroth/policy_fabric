# Proposal: a callback before Recipe submission in ProdEnv

**Status:** Implemented locally as `PolicyProdEnv` in this example. Potential
future NVFlare core contribution; no NVFlare core patch, issue, or PR is included.

## Problem and intended user experience

Policy approval must bind the finalized application files that NVFlare will
submit. Today, `ProdEnv` exports a Recipe's `FedJob` into a temporary directory
and immediately submits it through `SessionManager`. There is no public callback
between those operations. Exporting manually for approval, then calling
`recipe.execute(ProdEnv(...))`, would export again and lose the prepared files.

The researcher should keep normal Recipes and `recipe.execute(env)`. A generic
hook can prepare the final exported directory immediately before submission.
Signing, hashing, and Policy Fabric integration remain outside NVFlare core.

## Implemented prototype

```python
from prepare_job import PolicyJobPreparation
from submission_hook import PolicyProdEnv

prepare = PolicyJobPreparation(
    startup_kit_location=admin_kit,
    mock_asset_dids=["did:pdo:asset-a", "did:pdo:asset-b"],
)
env = PolicyProdEnv(
    startup_kit_location=admin_kit,
    username="admin@example.org",
    before_submit=prepare,
)
run = recipe.execute(env)
result_directory = run.get_result(timeout=180)
```

`PolicyProdEnv` inherits `ProdEnv.deploy()`, monitoring, abort, and result retrieval.
Its name describes this integration's purpose; the `before_submit` callback is
generic, with Policy Fabric behavior supplied by `PolicyJobPreparation`.
It replaces only the environment's private session-manager instance with a
subclass that adds the callback to the export/submission lifecycle. No installed
package is modified, and no job or Recipe method is monkey-patched.

The prototype duplicates the short `SessionManager.submit_job()` lifecycle and
uses private `_get_session_manager()` wiring. It is tested with NVFlare 2.9.0;
that coupling should be removed if the hook becomes a public core API.

## Callback contract

```python
before_submit: Callable[[pathlib.Path], None] | None
```

| Property | Behavior |
| --- | --- |
| Timing | Once per submission attempt, after Recipe finalization/environment processing and job export, before submission. |
| Argument | Path to the exported **job directory**, including `meta.json` and applications, not the parent export directory. |
| Mutation | Synchronous edits in place. The callback must return only after all work is complete. |
| Success | Return `None`. Submission uses that same directory without another Recipe export. |
| Failure | Raise an exception. Non-`None` returns, including `False`, are errors. No submission session is opened after a hook failure. `ProdEnv.deploy()` wraps failures in `RuntimeError` with their cause. |
| Cleanup | The temporary directory is removed after success or failure. An opened session is closed even when submission raises. |
| No callback | Delegate to NVFlare's original `SessionManager` path. |
| Export-only use | `recipe.export(...)` and the Recipe `--export` mode do not submit and do not run this hook. |
| Repeated execution | Each attempt gets a fresh export and callback invocation. There is no automatic approval reuse or retry. |
| Ownership | Submitter-installed Python code; never a callback loaded from a received job. It is not a replacement for site authorization. |

The normal secret scan runs after the callback so newly added configuration is
included. It is read-only and does not modify signed files. Mock capability
strings can trigger the existing high-entropy warning; the example preserves
that scanner. Do not add plaintext credentials to job metadata.

The callback may copy its final files to a separate audit directory before
returning. The demo uses this to retain `exports/<case>/`. It must not retain the
temporary path for later work or mutate files asynchronously. The prototype
does not guarantee concurrent use of a stateful callback instance.

## Policy Fabric callback

`PolicyJobPreparation` performs these steps inside the hook:

1. Sign application contents with the submitter's startup-kit credentials.
2. Freeze each client application as `policy_payload.zip` and compute SHA-256.
3. Mint explicitly requested mock capabilities, or call a supplied
   `capability_provider({app_name: digest})` and attach the returned
   `{asset_did: {app_name: opaque_capability}}` bundle to job metadata.
4. Sign the final submission, including the archives and capability metadata.

A provider exception prevents submission. `PolicyFabricCapabilityProvider` now
connects to the installed upstream client and requests capabilities through its
inference runner. The deployment client supports the real PDO decision contract
and an explicitly selected mock mode. The site independently measures and checks
the application actually received.

A trusted Script credential must cover the exact frozen ZIP digest. For approval
that cannot complete inside this callback, use the separate `freeze()` and
`authorize()` preparation steps and submit the preserved export through the
normal Flare API. See the [approval workflow](README.md#approve-the-exact-archive-before-submitting).

## Candidate NVFlare core change

The proposed upstream interface would be **`ProdEnv(..., before_submit=callback)`**.
This keyword does **not** exist in stock NVFlare 2.9.0; the example currently
requires `PolicyProdEnv`.

A small core change could:

- Add an optional, default-`None` callback to `ProdEnv` and pass it to
  `SessionManager` separately from the Flare API session parameters.
- Invoke it immediately after `job.export_job()` and before scanning/submitting.
- Preserve existing behavior when omitted and document the lifecycle above.
- Test successful mutation, preparation failure, submission failure/cleanup,
  normal behavior without a callback, repeated execution, and export-only use.

Whether to expose the same hook on `PocEnv` can be discussed separately.
Policy-specific operations, capability schemas, the Guardian client, and frozen
archive packaging would remain in this example/integration package.

Before an upstream PR, open or identify an NVFlare issue and agree the public
name, `Path` argument type, signing order, and callback error semantics with the
maintainers. This document records a candidate API, not an accepted upstream plan.

## Evidence

See [tests/test_submission_hook.py](tests/test_submission_hook.py) for the
contract tests and [VALIDATION.md](VALIDATION.md) for recorded end-to-end results.
The secure demo submits all four cases through `Recipe.execute(PolicyProdEnv)`;
the approved case also retrieves its model through the inherited `Run` API.

Reviewed source: [ProdEnv](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/recipe/prod_env.py)
and [SessionManager](https://github.com/NVIDIA/NVFlare/blob/main/nvflare/recipe/session_mgr.py),
checked on 2026-11-06. The local compatibility implementation is
[submission_hook.py](submission_hook.py).
