# H100 validation — 2026-11-16

**Passed:** secure NVFlare training and aggregation, final MONAI segmentation
inference, and all three authorization rejection cases. The dedicated run
exited with code **0**. The final focused test suite passed **64 tests**.

## Runtime and sources

| Item | Validated value |
| --- | --- |
| Node | `r1u14`, accessed through `hroth@172.16.1.14` |
| GPUs | H100 NVL, one per site: devices 0 and 1 |
| Environment | Dedicated Python 3.11.15 venv; no global package changes |
| NVFlare | Latest release at validation: **2.9.1**, tag commit `5f2250f1cd63ca2b3aff1952d950e3c3e9af5496` |
| MONAI / PyTorch | 1.4.0 / 2.6.0+cu124, CUDA 12.4 |
| Bundle | `MONAI/brats_mri_segmentation`, published version **0.5.4** |
| Dataset | Official MSD Task01 BrainTumour; archive MD5 `240a19d752f0d9e9101544901065d872` verified |
| Policy Fabric integration | `feat/nvflare-integration`, base commit `4e137ff0352d98651492dadaf37b3729f57f852a`, plus this example |
| Policy service mode | Explicit mock issuance and unchanged upstream mock Guardians |

The [provenance record](validation/provenance.json) contains SHA-256 hashes of
the source files used by the run and downloaded bundle files. The additional
packaging regression test and documentation were completed after this run;
the executed runtime source files are unchanged. The integration remains
based on upstream `authorities@31776f3`; this work does not change that base
or validate the complete live PDO stack.

Starting points:

- [NVFlare 2.9.1 release](https://github.com/NVIDIA/NVFlare/releases/tag/2.9.1)
- [Released MONAI spleen Recipe and Client API example](https://github.com/NVIDIA/NVFlare/tree/2.9.1/examples/advanced/monai/spleen_ct_segmentation)
- [Published BraTS bundle 0.5.4](https://huggingface.co/MONAI/brats_mri_segmentation/tree/0.5.4)

## Training and inference

Two sites completed two FedAvg rounds with one local epoch per round. Each
site had four training and two validation subjects, with no overlap between
sites or between training and validation. Each site executed eight optimizer
steps in total; all 16 steps used the original bundle crop and augmentation.
The [split record](validation/split_summary.json) lists the subjects and seed.

The final aggregated checkpoint is **18,834,653 bytes** and differs from the
pretrained weights by an L2 norm of **0.396349**. It was evaluated on all four
held-out subjects, producing 240 × 240 × 155 NIfTI predictions in the original
image geometry.

| Region | Final mean Dice |
| --- | ---: |
| Tumor core (TC) | 0.7572 |
| Whole tumor (WT) | 0.8655 |
| Enhancing tumor (ET) | 0.7172 |
| Mean across regions | **0.7800** |

These are functional integration results from a small pilot. The pretrained
BraTS bundle may overlap MSD subjects, so this is **not an independent accuracy
benchmark** or evidence of convergence. Per-case and per-round metrics are in
[results.json](validation/results.json). Round metrics evaluate the incoming
global model; the table above evaluates the final aggregate after both rounds.

## Authorization checks

| Case | Job ID | Result | Client executions |
| --- | --- | --- | ---: |
| Allowed | `a31cd8c8-6b10-4728-9268-c35eb93e2a85` | `FINISHED:COMPLETED` | 2 |
| Tampered | `d619f7fc-ab2c-463a-ba34-46ef48c2e756` | `FINISHED:FAILED_TO_RUN`; Guardian HTTP 422 | 0 |
| Missing capability | `94bb5bdc-84ca-43d9-bbce-f3e325f3ff90` | `FINISHED:CAN_NOT_SCHEDULE` | 0 |
| Guardian unavailable | `c5a64816-1b54-4b37-9664-2f1044422109` | `FINISHED:FAILED_TO_RUN` | 0 |

The test verifies registration metadata for both site asset DIDs, two completed
rounds per site, execution from the deployed `custom/client.py`, final weight
changes, finite Dice metrics, and zero application-start markers for rejected
jobs. Tampering happens after capability issuance and is re-signed by the test
submitter, so that rejection specifically exercises the Guardian digest check.

Mock capabilities are forgeable. These checks validate enforcement behavior;
they do not establish a trusted issuer, encrypted capability verification,
credential validity, revocation, or replay protection.

## Reproduce and inspect

Dedicated H100 root:

```text
/scratch/hroth/Code/nvflare/policy-fabric-brats-20261116-jjiBqg
```

Successful command, run from that root:

```bash
.venv/bin/python -u repo/tools/dev/nvflare_poc/monai_brats/run_e2e.py \
  --output run04 --bundle-root bundles/brats_mri_segmentation \
  --dataset-root data/Task01_BrainTumour --splits splits
```

- `logs/run04.log`: harness output; `run04.exit`: exit code.
- `logs/tests-final.log`: 64 passing tests; four expected secret-scan warnings
  about the base64 mock capabilities in existing submission-hook tests.
- `run04/logs/`: server, clients and Guardians.
- `run04/results.json`, `run04/provenance.json`, `run04/pip-freeze.txt`: evidence.
- `run04/exports/`: exact final signed submissions for each case.
- `run04/executed/`: application markers and per-site metric/step reports.
- `run04/evaluation/`: four predicted label volumes and evaluation JSON.
- `run04/workspace/policy-fabric-poc/prod_00/admin@example.org/transfer/`:
  downloaded final workspace containing `app_server/FL_global_model.pt`.

Only compact JSON evidence is copied into Git. Data, model weights, generated
private keys and full run workspaces remain on the H100. The run's federation
and Guardian processes were stopped and checked after completion.

## Integration details learned during validation

1. **Data conventions need explicit conversion.** MSD image channel order and
   tumor labels differ from those expected by this bundle. The example checks
   metadata and applies the conversion before the original bundle transforms.
2. **Native component policy still applies.** NVFlare 2.9.1's provisioned server
   allow-list needs the specific PyTorch persistor and SegResNet classes added.
   The example preserves the existing entries and keeps enforcement enabled.
3. **Portable checkpoint paths matter.** A relative local `initial_ckpt` is
   bundled by the Recipe. An absolute path is interpreted as a server-local
   file and would depend on the researcher's filesystem being shared.
4. **Execute the approved copy.** In this release, the in-process executor
   retained an absolute script path from the initial Recipe. The example uses
   the supported external-process mode, which executes the deployed archive's
   `custom/client.py`. A regression test checks both script and checkpoint
   packaging; the live run checks the actual execution path.

The first three setup attempts are preserved as `run01`–`run03` for diagnosis;
they do not contribute to the reported metrics. No installed NVFlare or MONAI
source was patched. Black and isort checks pass for the new Python files.
