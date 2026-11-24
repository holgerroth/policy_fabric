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

"""Run an isolated, secure, two-client NVFlare federation with the upstream mock Guardian."""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import nvflare
from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.lighter.constants import CtxKey
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import load_private_key_file, sign_folders
from nvflare.recipe import set_per_site_config
from pf_nvflare import CAPABILITIES, PAYLOAD_FILE
from prepare_job import PolicyJobPreparation, freeze_app
from submission_hook import PolicyProdEnv

HERE = Path(__file__).resolve().parent
ASSETS = {"site_a": "did:pdo:asset-a", "site_b": "did:pdo:asset-b"}
ADMIN = "admin@example.org"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def free_ports(count):
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def configure(output, ports):
    project = {
        "api_version": 3,
        "name": "policy-fabric-poc",
        "description": "Local mock Guardian integration; NOT production governance",
        "participants": [
            {"name": "localhost", "type": "server", "org": "poc", "fed_learn_port": ports[0], "admin_port": ports[1]},
            *[{"name": site, "type": "client", "org": "poc"} for site in ASSETS],
            {"name": ADMIN, "type": "admin", "org": "poc", "role": "project_admin"},
        ],
    }
    write_json(output / "project.json", project)
    ctx = Provisioner(
        str(output / "workspace"),
        [WorkspaceBuilder(), StaticFileBuilder(config_folder="config"), CertBuilder(), SignatureBuilder()],
    ).provision(prepare_project(project))
    if ctx.get(CtxKey.BUILD_ERROR):
        raise RuntimeError("NVFlare provisioning failed")
    kits = Path(ctx[CtxKey.CURRENT_PROD_DIR])
    for site in ["localhost", *ASSETS]:
        local = kits / site / "local"
        resources = json.loads((local / "resources.json.default").read_text())
        components = resources["components"]
        if site == "localhost":
            for component in components:
                if component["id"] == "job_manager":
                    component["path"] = "pf_nvflare.FrozenAppJobManager"
                    component["args"]["uri_root"] = str(output / "jobs-storage")
                if component["id"] == "job_scheduler":
                    component["args"].update(max_jobs=1, max_schedule_count=2, min_schedule_interval=1.0)
            resources["snapshot_persistor"]["args"]["storage"]["args"]["root_dir"] = str(output / "snapshots")
            components.append({"id": "policy_router", "path": "pf_nvflare.PolicyRegistrationRouter"})
        else:
            resources["client"]["site_config"] = {"asset_did": ASSETS[site]}
            port = ports[2 + list(ASSETS).index(site)]
            components.extend(
                [
                    {"id": "policy_security", "path": "pf_nvflare.PolicySecurityHandler"},
                    {
                        "id": "app_deployer",
                        "path": "pf_nvflare.PolicyAppDeployer",
                        "args": {"guardian_url": f"http://127.0.0.1:{port}", "timeout": 2.0, "guardian_mode": "mock"},
                    },
                ]
            )
        write_json(local / "resources.json", resources)
    return kits


def make_recipe(case):
    recipe = NumpyFedAvgRecipe(
        name=case,
        min_clients=2,
        num_rounds=1,
        model=[[1.0, 2.0], [3.0, 4.0]],
        train_script=str(HERE / "client.py"),
        launch_external_process=True,
    )
    set_per_site_config(recipe, {site: {} for site in ASSETS})
    return recipe


def prepare_case(job_dir, output, kits, case, preparation):
    preparation(job_dir)
    startup = kits / ADMIN / "startup"
    signing_key = load_private_key_file(str(startup / "client.key"))
    if case == "tampered":
        for app in preparation.digests:
            # The submitter can legitimately re-sign changed code, but cannot
            # update its prior Guardian approval. Isolate that enforcement layer.
            with tempfile.TemporaryDirectory() as staging:
                with zipfile.ZipFile(job_dir / app / PAYLOAD_FILE) as archive:
                    archive.extractall(staging)  # only this harness's own generated archive
                Path(staging, "tampered.txt").write_text("added AFTER capability issuance")
                sign_folders(staging, signing_key, str(startup / "client.crt"))
                (job_dir / app / PAYLOAD_FILE).write_bytes(freeze_app(Path(staging)))
    elif case == "missing-capability":
        meta_path = job_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["custom_props"][CAPABILITIES].pop(ASSETS["site_b"])
        write_json(meta_path, meta)
    sign_folders(str(job_dir), signing_key, str(startup / "client.crt"))
    # Keep the hook's final submission for inspection without exporting again.
    shutil.copytree(job_dir, output / "exports" / case)


def stop_process(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run(output, timeout):
    # Refuse to overwrite a previous run or any arbitrary existing directory.
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)  # generated startup kits contain test private keys
    logs = output / "logs"
    logs.mkdir()
    markers = output / "executed"
    markers.mkdir()
    ports = free_ports(4)
    kits = configure(output, ports)
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    env["PF_POC_EXECUTION_MARKERS"] = str(markers)
    processes = []
    handles = []
    session = None
    results = {
        "nvflare_version": nvflare.__version__,
        "mock_guardian": True,
        "submission": "Recipe.execute(PolicyProdEnv(before_submit=...))",
        "cases": {},
    }

    def launch(name, args, cwd=output):
        handle = (logs / f"{name}.log").open("w")
        handles.append(handle)
        process = subprocess.Popen(
            [sys.executable, "-u", *args],
            cwd=cwd,
            env=env,
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
                    str(HERE.parent / "mock_inference_guardian" / "server.py"),
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
        for site in ASSETS:
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
            )
        print(f"Starting secure NVFlare {nvflare.__version__}; logs: {logs}", flush=True)
        session = new_secure_session(ADMIN, str(kits / ADMIN), timeout=60)
        deadline = time.monotonic() + 60
        while {client.name for client in session.get_system_info().client_info} != set(ASSETS):
            if time.monotonic() > deadline or any(p.poll() is not None for p in processes):
                raise RuntimeError(f"clients did not register; inspect {logs}")
            time.sleep(1)
        print("Both clients registered. Submitting positive and negative cases.", flush=True)
        for case in ("allowed", "tampered", "missing-capability", "guardian-unavailable"):
            preparation = PolicyJobPreparation(kits / ADMIN, mock_asset_dids=ASSETS.values())

            def before_submit(job_dir):
                prepare_case(job_dir, output, kits, case, preparation)

            prod_env = PolicyProdEnv(
                startup_kit_location=str(kits / ADMIN),
                username=ADMIN,
                login_timeout=60,
                before_submit=before_submit,
            )
            if case == "guardian-unavailable":
                for guardian in guardians:
                    stop_process(guardian)
            execution = make_recipe(case).execute(prod_env)
            job_id = execution.get_job_id()
            print(f"{case}: submitted {job_id}", flush=True)
            rc, meta = session.monitor_job_and_return_job_meta(job_id, timeout=timeout, poll_interval=1)
            if meta is None:
                raise RuntimeError(f"{case}: job monitoring ended without result ({rc})")
            status = meta["status"]
            executed = sorted(path.name for path in markers.glob(f"{job_id}-*.txt"))
            result = {
                "job_id": job_id,
                "status": status,
                "executed": executed,
                "approved_digests": preparation.digests,
                "deploy_detail": meta.get("job_deploy_detail"),
                "schedule_history": meta.get("schedule_history"),
            }
            if case == "allowed":
                # Runtime client metadata is propagated into deployed workspaces;
                # it is not necessarily persisted in the job-store metadata API.
                deployed_meta = json.loads((kits / "site_a" / job_id / "meta.json").read_text())
                registered = deployed_meta.get("job_clients", [])
                result["registered_clients"] = registered
                assert status == "FINISHED:COMPLETED" and len(executed) == 2, result
                executed_paths = {site: (markers / f"{job_id}-{site}.txt").read_text().strip() for site in ASSETS}
                for site, script_path in executed_paths.items():
                    assert Path(script_path) == kits / site / job_id / f"app_{site}" / "custom/client.py", script_path
                result["executed_paths"] = executed_paths
                # Verify actual registration propagation, not just configured local values.
                observed = {client["name"]: client.get("site_config", {}).get("asset_did") for client in registered}
                assert observed == ASSETS, observed
                result_path = execution.get_result(timeout=timeout)
                assert result_path, "Recipe Run did not return a result workspace"
                models = list(Path(result_path).rglob("server.npy"))
                assert len(models) == 1, models
                model = np.load(models[0], allow_pickle=False)
                np.testing.assert_array_equal(model, [[2.0, 3.0], [4.0, 5.0]])
                result["final_model"] = model.tolist()
                result["result_path"] = result_path
            else:
                assert status in ("FINISHED:FAILED_TO_RUN", "FINISHED:CAN_NOT_SCHEDULE"), result
                assert not executed, result
                details = json.dumps(result)
                expected = {
                    "tampered": "Guardian denied request: HTTP 422",
                    "missing-capability": "Policy Fabric: missing capability for this site",
                    "guardian-unavailable": "Guardian unavailable",
                }[case]
                assert expected in details, result
            results["cases"][case] = result
            write_json(output / "results.json", results)
            print(f"{case}: PASS — {status}, {len(executed)} client executions", flush=True)
        results["passed"] = True
        write_json(output / "results.json", results)
        print(f"All cases passed: {output / 'results.json'}", flush=True)
    finally:
        if session:
            try:
                session.shutdown_system()
            except Exception as exc:
                print(f"Shutdown API: {type(exc).__name__}; stopping only this run's child processes.", flush=True)
            session.close()
        for process in reversed(processes):
            stop_process(process)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories are refused")
    parser.add_argument("--timeout", type=float, default=180, help="Maximum seconds per job")
    args = parser.parse_args()
    run(args.output.resolve(), args.timeout)
