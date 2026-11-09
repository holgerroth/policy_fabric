# Clarify the Guardian contract for FL job authorization and local data access

**Status:** Filed on 2026-11-03 as
[hasan7n/tmp-policies#10](https://github.com/hasan7n/tmp-policies/issues/10).
**Target:** [hasan7n/tmp-policies](https://github.com/hasan7n/tmp-policies/issues).
**Source check:** 2026-11-03. The mock server, README, and generator on both
`main` and `inference` match the versions used by this PoC.

**Historical record:** The text below describes the issue as filed. As of
2026-11-09, the new MLCommons repository's `authorities` branch includes the
decision-only inference operation. The migrated integration uses that operation;
this document is not a description of the current contract.

---

We have an [NVFlare integration PoC](https://github.com/holgerroth/tmp-policies/tree/feat/nvflare-policyfactory-poc/tools/dev/nvflare_poc)
working with the existing mock inference Guardian **without modifying it**.
Each NVFlare client hashes the exact serialized application it receives, sends
that digest and an opaque capability to its local Guardian, and permits
deployment only after an explicit matching allow response. NVFlare performs
the computation and enforces the decision; the Guardian does not execute the job.

The remaining interface question is how to move from this mock to a real Guardian.

## Current behavior

- The [mock Guardian](https://github.com/hasan7n/tmp-policies/blob/inference/tools/dev/mock_inference_guardian/server.py)
  accepts `POST /process_capability`, compares the capability's `script_digest`
  with `request_context.calculated_script_digest`, and returns a JSON object
  containing `allowed`, `script_digest`, `method_name`, and `mock` on success.
  A missing or mismatched digest returns HTTP 422. The success body currently
  uses `application/octet-stream`.
- The [real inference operation](https://github.com/hasan7n/tmp-policies/blob/inference/tools/guardians/inference/guardian_core/inference.py)
  returns the asset data after a matching digest check. Our current deployment
  gate expects an allow response and does not consume data. This is a choice
  in our PoC, not a requirement of NVFlare itself.

The mock capabilities are base64 JSON, not signed or encrypted. These tests
demonstrate integration and enforcement, not trusted policy approval. Our
current client deliberately requires `mock: true`; connecting the real Guardian
will require an explicit client update as well as an agreed Guardian contract.

## Integration options to discuss

Could we clarify whether the intended FL integration should authorize job
execution, mediate access to the local dataset, or support both?

- **Authorization decision:** the Guardian returns allow/deny; NVFlare enforces
  it while the training application uses the site's existing data access path.
  This fits the current PoC and avoids moving dataset contents through the
  deployment gate. A real Guardian operation with this contract would need to
  be agreed if we choose this approach.
- **Authorized local data release:** retain the real inference operation's
  model. Successful capability redemption supplies the input data to the local
  FL client, which passes it to the approved application. The mock could offer
  a sample-data response to exercise this path. Our integration would need a
  defined response contract and a controlled data handoff to the job process.
  A separate `allowed` field is not inherently necessary if successful
  redemption and denial are unambiguous in the agreed protocol.

Data release can make the Guardian the access gate for a protected dataset,
provided jobs cannot bypass it through another filesystem or storage path.
Returning data alone does not establish that isolation or prevent reuse after
release. Data stays on the data owner's machine in this model.

## Shared requirements

- Keep the input framework-independent: an opaque capability and a measured
  `sha256:<hex>` digest. Document that the existing `script_digest` field can
  name a serialized application archive. No job parser or manifest is needed
  in the Guardian.
- Document the chosen operation, response format, success/denial semantics,
  and how test responses differ from real capability validation. Preserve the
  existing data-release behavior.
- Keep measurement and execution enforcement in the site-installed FL
  integration, and capability validation in the Guardian. The reported hash
  is not remote attestation. Policy/asset and Guardian binding must come from
  the capability contract, not just a matching hash or routing metadata.
- Exercise matching-digest, mismatched-digest, and malformed-input cases;
  verify rejection of invalid capabilities on the real path. Denial, timeout,
  or an invalid response must prevent execution. If data release is chosen,
  also demonstrate the approved application's use of the local sample data
  and denial of release for a rejected request.

No mock changes are needed to continue the current PoC. This issue would agree
the intended boundary before prescribing a new API or changing the mock.
