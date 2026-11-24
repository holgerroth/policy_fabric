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

"""Ensure exported jobs execute packaged code and include the pretrained checkpoint."""

import json
import subprocess
import sys
from pathlib import Path

from demo import ADMIN, configure, free_ports
from job import HERE, build_recipe, package_bundle
from job_hash import hash_file


def make_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "configs").mkdir(parents=True)
    (bundle / "models").mkdir()
    (bundle / "configs/metadata.json").write_text(json.dumps({"version": "0.5.4"}))
    (bundle / "configs/train.json").write_text(
        json.dumps({"network_def": {"_target_": "SegResNet", "in_channels": 4, "out_channels": 3}})
    )
    checkpoint = bundle / "models/model.pt"
    checkpoint.write_bytes(b"checkpoint-export-sentinel")
    return bundle


def test_portable_job_contains_code_and_checkpoint(tmp_path):
    bundle = make_bundle(tmp_path)
    checkpoint = bundle / "models/model.pt"
    recipe = build_recipe(bundle)
    export = tmp_path / "exports"
    recipe.export(str(export))
    job_dir = export / "monai_brats"
    package_bundle(job_dir, bundle)
    config = json.loads((job_dir / "app_server/config/config_fed_server.json").read_text())
    persistor = next(c for c in config["components"] if c["id"] == "persistor")
    source = Path(persistor["args"]["source_ckpt_file_full_name"])
    assert not source.is_absolute()
    assert (job_dir / "app_server/custom" / source).read_bytes() == checkpoint.read_bytes()
    for site in ("site_a", "site_b"):
        app = job_dir / f"app_{site}"
        config = json.loads((app / "config/config_fed_client.json").read_text())
        executor = config["executors"][0]["executor"]["args"]
        assert executor["execution_mode"] == "external_process"
        assert "custom/client.py" in executor["command"]
        for name in ("client.py", "msd_transforms.py", "bundle_config.py", "bundle/configs/train.json"):
            assert (app / "custom" / name).is_file()


def test_freeze_cli_honors_destination_and_refuses_overwrite(tmp_path):
    bundle = make_bundle(tmp_path)
    workspace = tmp_path / "provisioning"
    workspace.mkdir()
    kits = configure(workspace, free_ports(4))
    destination = tmp_path / "approval"
    command = [
        sys.executable,
        str(HERE / "job.py"),
        "--bundle-root",
        str(bundle),
        "--startup-kit",
        str(kits / ADMIN),
        "--freeze-only",
        "--freeze-dir",
        str(destination),
    ]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    job_dir = destination / "monai_brats"
    assert report["job_dir"] == str(job_dir)
    measured = {app: hash_file(job_dir / app / "policy_payload.zip") for app in report["digests"]}
    assert measured == report["digests"] and len(measured) == 2
    assert not (tmp_path / "exports").exists()
    again = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert again.returncode != 0 and "Export already exists" in again.stderr
    assert measured == {app: hash_file(job_dir / app / "policy_payload.zip") for app in measured}
