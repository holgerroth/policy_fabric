# Policy Fabric × NVFlare: job authorization integration

A standard NVFlare Recipe prepares a frozen client application, requests a
capability through the installed Policy Fabric client, and submits through the
normal Flare API. Each site's NVFlare deployment plugin measures the received
archive and asks its configured Guardian whether it may execute it.

This example is based on MLCommons Policy Fabric's **`authorities` branch at
[`31776f3`](https://github.com/mlcommons/policy_fabric/commit/31776f3df799047d001067271a1155e21bda1cc4)**
(November 8, 2026). That branch includes the `inference-guardian` changes and the
new request-digest/authority APIs; these are ahead of upstream `main`.

[Design and team schematic](DESIGN.md) · [Validation evidence](VALIDATION.md) ·
[Potential NVFlare core hook](NVFLARE_HOOK_PROPOSAL.md)

## What is implemented

| Piece | Implementation |
| --- | --- |
| Researcher-side Policy Fabric connection | `PolicyFabricCapabilityProvider` calls the existing registry and `InferenceGuardianActionRunner.mint_capability()`, including readiness and policy evaluation. |
| Capability routing | NVFlare registration carries `client.site_config.asset_did`; the server reads `Client.get_site_config()`. |
| Guardian connection | `GuardianClient(mode="pdo")` uses the real inference Guardian's envelope/decision contract. Explicit `mode="mock"` retains local testing. |
| Recipe submission | Example-local `PolicyProdEnv(before_submit=...)`; preparation runs after export and before standard submission. |
| Site enforcement | Job-level authorization checks metadata; `PolicyAppDeployer` checks the actual archive before normal deployment. |
| Hashing | Generic SHA-256 over exact bytes. `FrozenAppJobManager` preserves those bytes through NVFlare storage. |

The local demo uses mock issuance and mock Guardians with a **real, secure
NVFlare federation**. Client/Guardian contract tests exercise the actual upstream
runner and operation with service dependencies stubbed. A complete PDO ledger,
trusted credential, encrypted capability and real Guardian deployment has **not
yet been validated with NVFlare**. No NVFlare core or upstream Guardian changes
are needed for this implementation.

## Run the self-contained demo

From the repository root, using Python 3.11 or newer:

```bash
python3 -m venv .venv-nvflare-poc
source .venv-nvflare-poc/bin/activate
python -m pip install -r tools/dev/nvflare_poc/requirements.txt
python -m pytest -q tools/dev/nvflare_poc/tests
python tools/dev/nvflare_poc/demo.py --output tools/dev/nvflare_poc/runs/demo-01
```

Use a new output directory. The harness provisions test certificates, starts
one server, two clients and two unchanged upstream mock Guardians, then shuts
down its own processes. It needs local sockets, but no GPU, Docker or dataset.
All four cases use `recipe.execute(PolicyProdEnv(...))`:

| Case | Expected result |
| --- | --- |
| `allowed` | Both clients execute one NumPy FedAvg round; model becomes `[[2, 3], [4, 5]]`. |
| `tampered` | Re-signed archive changed after approval; Guardian HTTP 422; no client script executes. |
| `missing-capability` | Site B's job-authorization hook denies scheduling; no client script executes. |
| `guardian-unavailable` | Deployment fails closed; no client script executes. |

`results.json`, logs, final exports, execution markers and result workspaces are
saved under the run directory. Generated startup kits contain test private keys;
run directories are ignored by Git. Mock capabilities are unsigned base64 JSON
and freely forgeable: the demo validates enforcement plumbing, not trusted approval.

## Recipe hook

With this directory on `PYTHONPATH`, use an ordinary Recipe and choose a provider:

```python
from policy_client import PolicyFabricCapabilityProvider
from prepare_job import PolicyJobPreparation
from submission_hook import PolicyProdEnv

provider = PolicyFabricCapabilityProvider.from_file("policy-client.json")
prepare = PolicyJobPreparation(admin_kit, capability_provider=provider)
env = PolicyProdEnv(
    startup_kit_location=admin_kit,
    username="admin@example.org",
    before_submit=prepare,
)
run = recipe.execute(env)
print(run.get_job_id())
print(run.get_result(timeout=180))
```

For the isolated mock federation, replace the provider with
`PolicyJobPreparation(admin_kit, mock_asset_dids=["did:pdo:asset-a", "did:pdo:asset-b"])`
and configure each deployer with `guardian_mode="mock"`.

The hook signs the application contents, freezes and hashes each client app,
obtains capabilities, attaches them to metadata, and signs the final exported
job. Failure prevents submission. Submission uses that directory without a
second export. Status, abort and results use the inherited `ProdEnv`/`Run` APIs.
`before_submit` is an example-local prototype, **not a stock NVFlare 2.9 API**.
The installed NVFlare package remains unchanged. Its normal secret scan remains
active and can flag the high-entropy capability strings.

## Connect the installed Policy Fabric client

Use the PDO/Django client environment from this checkout, bootstrapped and
configured as described in the [client setup](../../pdo_client/webapp/README.md)
and [onboarding guide](../../../onboarding.md). Install this example's NVFlare
requirements in that environment. The small standalone demo environment does
not install PDO. The bridge loads `tools/pdo_client/webapp` by default; an
explicit `webapp_dir` in its JSON configuration can point to that installed copy.
It is a bridge to the current repository's Python modules, not a stable public SDK.

The calling process needs the **same exported settings** as the initialized
client: `PDO_INSTALL_ROOT`, `PDO_CONTRACTS_ROOT`, `PDO_LEDGER_URL`,
`PDO_LEDGER_TYPE`, `F_SERVICE_HOST`, `LEDGER_CERT_PATH`, `SITE_TOML_SOURCE`,
`USER_KEYS_FOLDER`, `PREFERRED_ESERVICE_URL`, `ASSET_REGISTRY_URL`,
`TEMPLATE_REGISTRY_URL`, `SCRATCH_DIR`, `PDO_LOG_FILE`, and `PDO_LOG_LEVEL`.
The upstream webapp launcher exports these in its own process; launching a new
shell does not inherit that process's settings. Activate the PDO environment and
supply the settings there too. Reuse the correct ledger, keys and bootstrapped
scratch state; install matching `pdo-contracts` revisions for this branch, as
specified in the onboarding guide.

Copy [policy-client.example.json](policy-client.example.json) and replace all
placeholders with existing identities. Each target names a data `asset_did`, an
NVFlare client `app_name`, and role wallets. `Script` is the **identity of the
approved application archive**, while `User` is the requester if the site's
policy requires it. Add `params.project` if required by policy. Do not put
`script_digest` in this config; it is always measured from the frozen archive.
Every frozen client app must have a target, and every participating data asset
must have its appropriate target.

The provider resolves asset metadata via `registry_client.get_asset_by_did()`,
uses `action_runners.build_runner()`, and calls the inference runner's
`mint_capability()`. That preserves upstream readiness checks and
`pdo_runner.create_capability(request={"script_digest": ...})`. A refusal aborts
the submission; there is no mock fallback. The encrypted capability envelope is
base64-encoded for metadata transport. Only the Guardian opens its operation.
The site's Guardian endpoint is configured locally, not taken from registry or
job metadata.

### Approve the exact archive before submitting

The new inference delivery policy requires a trusted **`ScriptHashCredential`**
whose `scriptHash` equals `request.script_digest`. A credential for a Python
source file does not authorize the ZIP containing that file. Signatures and ZIP
metadata affect the digest too. A fresh Recipe execution can produce new bytes;
never assume that an earlier approval covers a re-export.

For an asynchronous code-review/approval process, use the two-stage preparation:

1. Export the Recipe once with `recipe.export("exports", env=env)` (where `env`
   has the intended production settings). Locate its job directory, then freeze:

   ```bash
   python tools/dev/nvflare_poc/prepare_job.py exports/my-job \
     --startup-kit /path/to/admin@example.org --freeze-only
   ```

2. Retain each `app_*/policy_payload.zip` and the printed `sha256:` digest.
   Use the Policy Fabric **Script identity** for that archive. Have a trusted
   authority review those bytes and issue the Script digest credential to that
   identity, following the [inference tutorial's authority flow](../../../docs/docs/tutorial_inference.md#22-sign-the-script-digest).
   Supply the other credentials required by the site's selected policy cards.
   For distinct app archives, use appropriately approved Script identities.

3. In the configured Policy Fabric client environment, mint capabilities and
   sign the same outer job without re-exporting or recreating its ZIPs:

   ```bash
   python tools/dev/nvflare_poc/prepare_job.py exports/my-job \
     --startup-kit /path/to/admin@example.org \
     --authorize-only --policy-client policy-client.json
   ```

4. Submit **that directory** with the standard NVFlare admin `submit_job` command
   or `Session.submit_job("exports/my-job")`. Calling `recipe.execute()` here
   would export again. Keep these files under the submitter's control until
   submission.

`PolicyJobPreparation.freeze(job_dir)` and `.authorize(job_dir)` expose the same
steps in Python. The single submission hook is appropriate when the installed
client/issuer can satisfy approval for the newly frozen bytes during that call.
The integration does not create trusted Script credentials or bypass review.
No Policy Fabric toy FL server or script-fetching flow is used to submit NVFlare jobs.

## Site and server configuration

Merge this fragment into each site's existing `local/resources.json`:

```json
{
  "client": {"site_config": {"asset_did": "did:pdo:replace-with-data-asset"}},
  "components": [
    {"id": "policy_security", "path": "pf_nvflare.PolicySecurityHandler"},
    {
      "id": "app_deployer",
      "path": "pf_nvflare.PolicyAppDeployer",
      "args": {
        "guardian_url": "http://127.0.0.1:7900",
        "guardian_mode": "pdo",
        "timeout": 5.0
      }
    }
  ]
}
```

On the server, add a `pf_nvflare.PolicyRegistrationRouter` component and change
the existing `job_manager` implementation to `pf_nvflare.FrozenAppJobManager`,
retaining its normal `uri_root` setting. [demo.py](demo.py) shows complete
provisioning/configuration. Install `pf_nvflare.py` and `job_hash.py` on the
server and clients' trusted Python paths, outside submitted jobs. Preserve
existing unrelated components and native site controls.

The default Guardian mode is `pdo`. It sends the decoded outer capability JSON
with `request_context.calculated_script_digest`; it does not decrypt the
operation. It requires HTTP 200, `allowed: true`, a matching `script_digest`,
`method_name: "do_inference"`, and no mock marker. Explicit mock mode uses the
mock's `capability_b64` wrapper and requires `mock: true`. Both fail closed on
errors, redirects, timeouts, ambiguous decisions and oversized responses.

This version supports only site-configured **loopback HTTP** Guardian origins.
Issuer and Guardian hosting/administration remain architectural decisions; a
remote Guardian would need an authenticated transport and explicit integration
changes. The upstream inference launcher also starts a toy FL client: that toy
client is not NVFlare. A real deployment must arrange a Guardian endpoint next
to the NVFlare daemon (or a suitable shared network namespace) instead.

## Hashing and identity boundaries

`job_hash.py` hashes any supplied file/bytes as `sha256:<hex>`, independently of
FL framework. No manifest or field mapping is involved. This integration measures
**each complete client application ZIP**, including packaged code, config and
signatures. It does not measure the server app, outer job metadata, external
datasets/dependencies, or later model/task messages.

NVFlare normally unpacks and repacks apps. `FrozenAppJobManager.get_app()` returns
the inner `policy_payload.zip` unchanged, so the client measures the approved
bytes before its normal deployer unpacks them. The inner archive is authoritative
for execution; adjacent exported files remain for native submission validation.

Capabilities live outside the hash in
`custom_props.policy_fabric_capabilities[asset_did][app_name]`. The server rebuilds
routes from authenticated registration; sites check their own configured DID.
An **asset DID is not a policy-agent ID or token ID**, and authenticated site
registration does not prove ownership of that asset. The registry and PDO policy
path resolve those other identities. This PoC assumes one configured data asset
and Guardian per site, a flat deployment map, and a trusted site daemon/plugin.
The complete bundle travels in shared metadata; routing is not per-site secrecy.

Policy Fabric owns credential/policy evaluation and capability verification.
NVFlare owns measurement, scheduling, execution and existing local controls.
Full capability trust, audience, expiry/replay/revocation behavior, deployment
isolation and dataset enforcement must be validated in the real PDO setup;
none is established by the mock run. See [the design](DESIGN.md) for details.
