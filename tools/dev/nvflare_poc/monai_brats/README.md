# MONAI BraTS segmentation with Policy Fabric authorization

This example runs the published **MONAI `brats_mri_segmentation` 0.5.4 bundle**
on MSD Task01 BrainTumour through a secure, two-site **NVFlare 2.9.1** federation.
It uses `FedAvgRecipe`, the Client API, and `MonaiAlgo`, following the released
[MONAI spleen example](https://github.com/NVIDIA/NVFlare/tree/2.9.1/examples/advanced/monai/spleen_ct_segmentation).
Submission uses the existing example-local `PolicyProdEnv` hook.

[Latest H100 rerun](../VALIDATION.md#local-guardian-rerun--2026-11-24) ·
[Original workload validation](VALIDATION.md) ·
[Latest machine-readable evidence](../validation/local-guardians-20261124.json)

The training, aggregation, TLS federation and job signatures are real. The
self-contained harness uses **mock capabilities and mock Guardians**; it does
not establish trusted Policy Fabric approval. See the parent
[integration design](../DESIGN.md) for that distinction and the real PDO bridge.

## Workload

- SegResNet with the bundle's pretrained weights, four MRI inputs, and three
  output regions: tumor core (TC), whole tumor (WT), enhancing tumor (ET).
- Two FedAvg rounds, one local epoch per round, batch size one. Defaults select
  four training and two validation subjects per site (12 disjoint subjects).
- Original bundle architecture, loss, random augmentation, training crop
  (224 × 224 × 144) and validation sliding window (240 × 240 × 160).
- Each site validates the incoming global model before local training. After
  aggregation, the harness evaluates the final model on the four validation
  subjects and saves full-volume NIfTI predictions.

The data conversion is necessary: MSD stores **FLAIR, T1w, T1gd, T2w**; the
bundle expects **T1c, T1, T2, FLAIR**. Images are reordered `[2, 1, 3, 0]`.
MSD labels are `1=edema`, `2=non-enhancing tumor`, `3=enhancing tumor`:
TC uses `{2,3}`, WT uses `{1,2,3}`, ET uses `{3}`. The bundle's stock MONAI 1.4
BraTS transform expects labels `1/2/4`, so it cannot be used unchanged.
Preparation checks the dataset metadata; unit tests check conversion and
patient separation. This conversion is part of the submitted, hashed code.

This is a functional pilot, not an independent accuracy benchmark. The bundle
was pretrained on BraTS data, which may overlap MSD subjects. The held-out
subjects are separate from this run's training subjects only.

## Run end to end

Use Linux, Python 3.11, and two GPUs with enough memory for the original bundle
crop. The recorded validation uses one H100 NVL per site. Run these commands
from the repository root. Keep data, environments and generated keys outside
the repository (or under the ignored parent `runs/` directory).

```bash
export BRATS_WORK=/scratch/$USER/policy-fabric-brats
mkdir -p "$BRATS_WORK"
python3.11 -m venv "$BRATS_WORK/.venv"
source "$BRATS_WORK/.venv/bin/activate"
python -m pip install -r tools/dev/nvflare_poc/monai_brats/requirements.txt

python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="MONAI/brats_mri_segmentation",
    revision="0.5.4",
    local_dir=os.path.join(os.environ["BRATS_WORK"], "bundle"),
    allow_patterns=["configs/*", "scripts/*", "docs/*", "models/model.pt", "LICENSE*"],
)
PY

python tools/dev/nvflare_poc/monai_brats/prepare_data.py \
  --download --dataset-root "$BRATS_WORK/data/Task01_BrainTumour" \
  --output "$BRATS_WORK/splits"

python -m pytest -q tools/dev/nvflare_poc/tests tools/dev/nvflare_poc/monai_brats
python tools/dev/nvflare_poc/monai_brats/run_e2e.py \
  --bundle-root "$BRATS_WORK/bundle" \
  --dataset-root "$BRATS_WORK/data/Task01_BrainTumour" \
  --splits "$BRATS_WORK/splits" --output "$BRATS_WORK/run01" --gpus 0 1
```

Use fresh split and run directories. Downloading the official archive takes
about 7.1 GiB; extraction requires additional space. Its MD5 is checked against
MONAI's published `240a19d752f0d9e9101544901065d872`. There are 484 labeled
subjects; the unlabeled test set is not used. Increase `--train-per-site` and
`--val-per-site` when preparing splits; `--train-per-site 0` uses the remaining
labeled subjects, balanced across sites. Increase `--rounds`, `--local-epochs`
and the per-job `--timeout` when running larger experiments.

The harness provisions a fresh local TLS federation, launches two clients and
one upstream mock Guardian per site on `127.0.0.1` (separate processes and ports),
and runs these checks. No external Guardian service is needed; see
[local hosting and real Guardian requirements](../README.md#local-guardian-hosting).

| Case | Required result |
| --- | --- |
| Allowed | Both sites complete training; final aggregate differs from pretrained weights; full-volume inference produces finite Dice metrics. |
| Tampered archive | Re-signed application differs from approved bytes; Guardian denies deployment; zero client executions. |
| Missing capability | Job authorization rejects the job before execution. |
| Guardian unavailable | Deployment fails closed; zero client executions. |

The server's native `class_allow_list` is extended with the exact
`nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor` and
`monai.networks.nets.SegResNet` classes. Its enforcement remains enabled.
No NVFlare core changes are required.

Outputs include `results.json`, `logs/`, signed `exports/`, `executed/` site
reports, a downloaded result workspace with `FL_global_model.pt`, and
`evaluation/` with per-region Dice and four prediction volumes. Test startup
kits contain private keys; do not commit run directories.

## Use an existing federation

Install the parent integration plugins in the server and client environments
as described in the [parent README](../README.md#site-and-server-configuration).
Install this directory's requirements on the researcher, server and clients.
Configure the server's two additional allowed classes as above. On each client,
set `PF_BRATS_DATA` to its local MSD directory and `PF_BRATS_SPLITS` to the
directory containing its `<site>.json` split before starting NVFlare. The example
targets `site_a` and `site_b`; data paths are local configuration, not job inputs.

For real approval, export and freeze the complete bundle application first:

```bash
python tools/dev/nvflare_poc/monai_brats/job.py \
  --bundle-root /path/to/brats_mri_segmentation \
  --startup-kit /path/to/admin@example.org \
  --freeze-only --freeze-dir /path/to/frozen-jobs
```

Have the trusted issuer approve each printed ZIP digest in its Script identity's
`ScriptHashCredential`, including the `sha256:` prefix and the other evidence
the site's policy requires. Then attach capabilities to that same export:

```bash
python tools/dev/nvflare_poc/prepare_job.py /path/to/frozen-jobs/monai_brats \
  --startup-kit /path/to/admin@example.org --authorize-only \
  --policy-client /path/to/policy-client.json
```

Submit that directory through the standard NVFlare admin `submit_job` command
or `Session.submit_job()`. Do not call `recipe.execute()` after approval: it
would re-export and may change the authorized bytes. Main's capability issuer
reads the digest from the Script credential, and the Guardian checks it against
the ZIP received by the site. It does not compare a request digest at issuance.

For one-call mock testing, run `job.py` with `--mock` and explicitly configured
mock Guardians. The real bridge requires the upstream PDO client environment
and trusted credentials; the complete live flow has not yet been validated.
The [parent README](../README.md#approve-the-exact-archive-before-submitting)
describes the approval lifecycle in detail.

`package_bundle()` copies the bundle configuration and helper code into each
client application. The Recipe bundles the pretrained checkpoint into the
server application by using a relative local checkpoint path. Its standard
external-process mode executes `custom/client.py` from the deployed archive;
the E2E check verifies that location.
The preparation hook then signs, freezes, hashes, and authorizes the client
archives before standard submission. The dataset stays on the sites. The
Guardian decision binds the exact client application bytes; it does **not**
cover external data, installed dependencies, or evolving global weights. Those
require separate policies and runtime controls.

The final evaluation in the self-contained harness can access both sites' test
data because they share one H100 machine. In a distributed deployment, run
validation locally at each site and collect metrics according to site policy.

## Files

| File | Purpose |
| --- | --- |
| [job.py](job.py) | Recipe and packaging before submission. |
| [client.py](client.py) | Client API loop using MONAI `MonaiAlgo`. |
| [bundle_config.py](bundle_config.py) | Data and runtime overrides for the original bundle. |
| [msd_transforms.py](msd_transforms.py) | Explicit MRI channel and label conversion. |
| [prepare_data.py](prepare_data.py) | Download verification and deterministic disjoint splits. |
| [run_e2e.py](run_e2e.py) | Secure federation and four authorization cases. |
| [evaluate.py](evaluate.py) | Final aggregate validation and NIfTI predictions. |
