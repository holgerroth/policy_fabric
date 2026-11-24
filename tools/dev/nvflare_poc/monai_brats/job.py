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

"""Standard MONAI FedAvg Recipe submitted through the Policy Fabric preparation hook."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import set_per_site_config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from policy_client import PolicyFabricCapabilityProvider
from prepare_job import PolicyJobPreparation
from submission_hook import PolicyProdEnv


def build_recipe(bundle_root, name="monai_brats", rounds=2, local_epochs=1):
    bundle_root = Path(bundle_root).resolve()
    metadata = json.loads((bundle_root / "configs/metadata.json").read_text())
    if metadata["version"] != "0.5.4":
        raise ValueError("This example is validated with brats_mri_segmentation bundle 0.5.4")
    network = json.loads((bundle_root / "configs/train.json").read_text())["network_def"]
    if network["_target_"] != "SegResNet":
        raise ValueError("Expected the original bundle SegResNet network")
    recipe = FedAvgRecipe(
        name=name,
        min_clients=2,
        num_rounds=rounds,
        model={
            "class_path": "monai.networks.nets.SegResNet",
            "args": {k: v for k, v in network.items() if k != "_target_"},
        },
        # 2.9.1 bundles relative local files; absolute paths refer to server files.
        initial_ckpt=os.path.relpath(bundle_root / "models/model.pt"),
        train_script=str(HERE / "client.py"),
        train_args=f"--local-epochs {local_epochs} --rounds {rounds}",
        # External execution uses custom/client.py from the deployed archive.
        launch_external_process=True,
        key_metric="val_mean_dice",
    )
    set_per_site_config(recipe, {"site_a": {}, "site_b": {}})
    recipe.add_server_config({"server": {"heart_beat_timeout": 600}})
    return recipe


def package_bundle(job_dir, bundle_root):
    """Include reviewed bundle code/config in every client archive before hashing.

    The Recipe packages the pretrained model in the server application.
    Data and site-specific data paths stay outside the submitted job.
    """
    job_dir, bundle_root = Path(job_dir), Path(bundle_root)
    meta = json.loads((job_dir / "meta.json").read_text())
    for app, sites in meta["deploy_map"].items():
        if sites == ["server"]:
            continue
        custom = job_dir / app / "custom"
        destination = custom / "bundle"
        destination.mkdir(parents=True, exist_ok=False)
        for folder in ("configs", "scripts", "docs"):
            if (bundle_root / folder).exists():
                shutil.copytree(bundle_root / folder, destination / folder)
        for path in bundle_root.glob("LICENSE*"):
            if path.is_file():
                shutil.copy2(path, destination / path.name)
        for module in ("bundle_config.py", "msd_transforms.py"):
            shutil.copy2(HERE / module, custom / module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--startup-kit", type=Path, required=True)
    parser.add_argument("--username", default="admin@example.org")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--freeze-dir", type=Path, default=Path("exports"), help="Destination for --freeze-only")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--freeze-only", action="store_true", help="Export and freeze for external digest approval")
    choice.add_argument("--mock", action="store_true", help="Use explicit insecure mock capabilities")
    choice.add_argument(
        "--policy-client", type=Path, help="Configured real Policy Fabric client (requires PDO runtime)"
    )
    args = parser.parse_args()
    provider = PolicyFabricCapabilityProvider.from_file(args.policy_client) if args.policy_client else None
    preparation = PolicyJobPreparation(
        args.startup_kit,
        mock_asset_dids=("did:pdo:asset-a", "did:pdo:asset-b") if args.mock else (),
        capability_provider=(lambda _: {}) if args.freeze_only else provider,
    )

    def before_submit(job_dir):
        package_bundle(job_dir, args.bundle_root)
        preparation(job_dir)

    env = PolicyProdEnv(startup_kit_location=str(args.startup_kit), username=args.username, before_submit=before_submit)
    recipe = build_recipe(args.bundle_root, rounds=args.rounds, local_epochs=args.local_epochs)
    if args.freeze_only:
        job_dir = args.freeze_dir.resolve() / "monai_brats"
        if job_dir.exists():
            parser.error(f"Export already exists: {job_dir}; use a fresh --freeze-dir")
        # Export does not call the submission hook or connect to the federation.
        recipe.export(str(args.freeze_dir.resolve()), env=env)
        package_bundle(job_dir, args.bundle_root)
        print(json.dumps({"job_dir": str(job_dir), "digests": preparation.freeze(job_dir)}, indent=2))
    else:
        run = recipe.execute(env)
        print(f"Job ID: {run.get_job_id()}")
        print(f"Result: {run.get_result(timeout=3600)}")
