# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Secure two-site H100 validation: MONAI training plus three Guardian rejection cases."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nvflare
from job import build_recipe, package_bundle
from nvflare.fuel.flare_api.flare_api import new_secure_session

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from demo import ADMIN, ASSETS, configure, free_ports, prepare_case, stop_process, write_json
from prepare_job import PolicyJobPreparation
from submission_hook import PolicyProdEnv


def run(output, bundle_root, dataset_root, splits, rounds=2, local_epochs=1, gpus=("0", "1"), timeout=1800):
    if nvflare.__version__ != "2.9.1":
        raise RuntimeError(f"Expected NVFlare 2.9.1, got {nvflare.__version__}")
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    logs, markers = output / "logs", output / "executed"
    logs.mkdir()
    markers.mkdir()
    ports = free_ports(4)
    kits = configure(output, ports)
    # NVFlare 2.9.1's native server policy requires these explicit additions.
    # Keep enforcement enabled and preserve the provisioned default list.
    resources_path = kits / "localhost/local/resources.json"
    resources = json.loads(resources_path.read_text())
    resources["class_allow_list"].extend(
        ["nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor", "monai.networks.nets.SegResNet"]
    )
    write_json(resources_path, resources)
    env = os.environ.copy()
    env.update(
        {
            "PATH": str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
            "PYTHONPATH": str(HERE.parent) + os.pathsep + env.get("PYTHONPATH", ""),
            "PF_BRATS_DATA": str(dataset_root),
            "PF_BRATS_SPLITS": str(splits),
            "PF_POC_EXECUTION_MARKERS": str(markers),
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
    )
    processes, handles = [], []
    session = None
    results = {
        "nvflare_version": nvflare.__version__,
        "bundle": "MONAI/brats_mri_segmentation@0.5.4",
        "mock_guardian": True,
        "submission": "Recipe.execute(PolicyProdEnv)",
        "rounds": rounds,
        "local_epochs": local_epochs,
        "gpus": list(gpus),
        "cases": {},
    }

    def launch(name, arguments, gpu=""):
        handle = (logs / f"{name}.log").open("w")
        handles.append(handle)
        process = subprocess.Popen(
            [sys.executable, "-u", *arguments],
            cwd=output,
            env={**env, "CUDA_VISIBLE_DEVICES": gpu},
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(process)
        return process

    try:
        guardians = [
            launch(
                f"guardian-{site}",
                [
                    str(HERE.parent.parent / "mock_inference_guardian/server.py"),
                    "--interface",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
            )
            for site, port in zip(ASSETS, ports[2:])
        ]
        launch(
            "server",
            [
                "-m",
                "nvflare.private.fed.app.server.server_train",
                "-m",
                str(kits / "localhost"),
                "-s",
                "fed_server.json",
                "--set",
                "secure_train=true",
                "org=poc",
                "config_folder=config",
            ],
        )
        for site, gpu in zip(ASSETS, gpus):
            launch(
                site,
                [
                    "-m",
                    "nvflare.private.fed.app.client.client_train",
                    "-m",
                    str(kits / site),
                    "-s",
                    "fed_client.json",
                    "--set",
                    "secure_train=true",
                    f"uid={site}",
                    "org=poc",
                    "config_folder=config",
                ],
                gpu=gpu,
            )
        session = new_secure_session(ADMIN, str(kits / ADMIN), timeout=90)
        deadline = time.monotonic() + 90
        while {client.name for client in session.get_system_info().client_info} != set(ASSETS):
            if time.monotonic() > deadline or any(p.poll() is not None for p in processes):
                raise RuntimeError(f"Clients failed to register; inspect {logs}")
            time.sleep(1)
        for case in ("allowed", "tampered", "missing-capability", "guardian-unavailable"):
            preparation = PolicyJobPreparation(kits / ADMIN, mock_asset_dids=ASSETS.values())

            def before_submit(job_dir):
                package_bundle(job_dir, bundle_root)
                prepare_case(job_dir, output, kits, case, preparation)

            if case == "guardian-unavailable":
                for guardian in guardians:
                    stop_process(guardian)
            recipe = build_recipe(bundle_root, name=f"monai_brats_{case}", rounds=rounds, local_epochs=local_epochs)
            execution = recipe.execute(
                PolicyProdEnv(
                    startup_kit_location=str(kits / ADMIN),
                    username=ADMIN,
                    login_timeout=90,
                    before_submit=before_submit,
                )
            )
            job_id = execution.get_job_id()
            print(f"{case}: submitted {job_id}", flush=True)
            _, meta = session.monitor_job_and_return_job_meta(job_id, timeout=timeout, poll_interval=2)
            if meta is None:
                raise RuntimeError(f"{case}: no terminal job metadata before timeout")
            executed = sorted(path.name for path in markers.glob(f"{job_id}-*.txt"))
            result = {
                "job_id": job_id,
                "status": meta["status"],
                "executed": executed,
                "approved_digests": preparation.digests,
                "deploy_detail": meta.get("job_deploy_detail"),
                "schedule_history": meta.get("schedule_history"),
            }
            if case == "allowed":
                assert meta["status"] == "FINISHED:COMPLETED" and len(executed) == 2, result
                client_reports = [json.loads((markers / f"{job_id}-{site}.json").read_text()) for site in ASSETS]
                assert all(len(report["rounds"]) == rounds for report in client_reports), client_reports
                for report in client_reports:
                    app_root = kits / report["site"] / job_id / f"app_{report['site']}"
                    assert Path(report["script_path"]).is_relative_to(app_root), report["script_path"]
                runtime_meta = json.loads((kits / "site_a" / job_id / "meta.json").read_text())
                registered = {
                    client["name"]: client.get("site_config", {}).get("asset_did")
                    for client in runtime_meta["job_clients"]
                }
                assert registered == ASSETS, registered
                result["registered_assets"] = registered
                result["clients"] = client_reports
                result_path = execution.get_result(timeout=timeout)
                assert result_path, "Run.get_result returned no workspace"
                checkpoints = list(Path(result_path).rglob("FL_global_model.pt"))
                assert len(checkpoints) == 1, checkpoints
                result["checkpoint"] = str(checkpoints[0])
                result["result_path"] = str(result_path)
            else:
                assert (
                    meta["status"] in ("FINISHED:FAILED_TO_RUN", "FINISHED:CAN_NOT_SCHEDULE") and not executed
                ), result
                expected = {
                    "tampered": "Guardian denied request: HTTP 422",
                    "missing-capability": "Policy Fabric: missing capability for this site",
                    "guardian-unavailable": "Guardian unavailable",
                }[case]
                assert expected in json.dumps(result), result
            results["cases"][case] = result
            write_json(output / "results.json", results)
            print(f"{case}: PASS ({meta['status']}, {len(executed)} client executions)", flush=True)
        from evaluate import evaluate

        results["final_evaluation"] = evaluate(
            bundle_root, Path(results["cases"]["allowed"]["checkpoint"]), dataset_root, splits, output / "evaluation"
        )
        results["passed"] = True
        write_json(output / "results.json", results)
        print(f"All training, inference and policy checks passed: {output / 'results.json'}", flush=True)
    finally:
        if session:
            try:
                session.shutdown_system()
            except Exception as exc:
                print(f"Shutdown API: {type(exc).__name__}; stopping only this run's processes.", flush=True)
            session.close()
        for process in reversed(processes):
            stop_process(process)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--gpus", nargs=2, default=["0", "1"])
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    run(
        args.output.resolve(),
        args.bundle_root.resolve(),
        args.dataset_root.resolve(),
        args.splits.resolve(),
        args.rounds,
        args.local_epochs,
        args.gpus,
        args.timeout,
    )
