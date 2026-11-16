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

"""MONAI bundle training via the standard NVFlare Client API."""

import argparse
import json
import math
import os
from pathlib import Path

import nvflare.client as flare
import torch
from bundle_config import overrides
from monai.fl.client import MonaiAlgo
from monai.fl.utils.constants import ExtraItems, FlStatistics
from monai.fl.utils.exchange_object import ExchangeObject
from nvflare.app_opt.monai import decomposers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.local_epochs < 1 or args.rounds < 1:
        parser.error("epochs and rounds must be positive")
    flare.init()
    decomposers.register_monai_types()
    if not torch.cuda.is_available():
        raise RuntimeError("This H100 PoC requires CUDA; refusing silent CPU fallback")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    site, job_id = flare.get_site_name(), flare.get_job_id()
    marker_root = Path(os.environ.get("PF_POC_EXECUTION_MARKERS", str(Path.cwd() / "monai_reports")))
    marker_root.mkdir(parents=True, exist_ok=True)
    (marker_root / f"{job_id}-{site}.txt").write_text("application started\n")
    bundle_root = Path(__file__).resolve().parent / "bundle"
    dataset_root = Path(os.environ["PF_BRATS_DATA"])
    datalist = Path(os.environ["PF_BRATS_SPLITS"]) / f"{site}.json"
    split = json.loads(datalist.read_text())
    report = {
        "site": site,
        "job_id": job_id,
        "script_path": str(Path(__file__).resolve()),
        "gpu": torch.cuda.get_device_name(0),
        "train_cases": len(split["training"]),
        "validation_cases": len(split["validation"]),
        "rounds": [],
    }
    report_path = marker_root / f"{job_id}-{site}.json"
    runtime = overrides(
        bundle_root, dataset_root, datalist, Path.cwd() / "monai_output", args.local_epochs * args.rounds
    )
    algo = MonaiAlgo(
        bundle_root=str(bundle_root),
        local_epochs=args.local_epochs,
        send_weight_diff=False,
        config_train_filename="configs/train.json",
        config_evaluate_filename=["configs/train.json", "configs/evaluate.json"],
        train_kwargs=runtime,
        eval_kwargs=runtime.copy(),
        disable_ckpt_loading=True,
    )
    algo.initialize(extra={ExtraItems.CLIENT_NAME: site, ExtraItems.LOGGING_FILE: False})
    try:
        while flare.is_running():
            incoming = flare.receive()
            exchange = ExchangeObject(weights=incoming.params)
            metrics = {key: float(value) for key, value in algo.evaluate(exchange).metrics.items()}
            if not metrics or not all(math.isfinite(value) and 0 <= value <= 1 for value in metrics.values()):
                raise RuntimeError(f"Invalid Dice metrics: {metrics}")
            algo.train(exchange)
            updated = algo.get_weights()
            steps = int(updated.statistics[FlStatistics.NUM_EXECUTED_ITERATIONS])
            if steps != args.local_epochs * len(split["training"]):
                raise RuntimeError(f"Unexpected number of training steps: {steps}")
            report["rounds"].append(
                {"round": incoming.current_round, "global_validation": metrics, "training_steps": steps}
            )
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            flare.send(flare.FLModel(params=updated.weights, metrics=metrics, meta={"NUM_STEPS_CURRENT_ROUND": steps}))
    finally:
        algo.finalize()


if __name__ == "__main__":
    main()
