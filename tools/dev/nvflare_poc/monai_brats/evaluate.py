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

"""Validate the final aggregated model and save full-volume MSD-space predictions."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from bundle_config import overrides
from monai.bundle import ConfigParser
from monai.data import Dataset, load_decathlon_datalist
from monai.metrics import compute_dice


def load_weights(path):
    # Inputs are the official downloaded bundle and this run's own checkpoint.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("model", checkpoint)
    return {key: torch.as_tensor(value) for key, value in weights.items()}


def evaluate(bundle_root, checkpoint, dataset_root, splits, output):
    bundle_root, output = Path(bundle_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("Final volume validation requires CUDA")
    parser = ConfigParser()
    parser.read_config(str(bundle_root / "configs/train.json"))
    parser.update(overrides(bundle_root, dataset_root, Path(splits) / "evaluation.json", output, 1))
    parser["bundle_root"] = str(bundle_root)
    network = parser.get_parsed_content("network_def").cuda().eval()
    weights = load_weights(checkpoint)
    initial = load_weights(bundle_root / "models/model.pt")
    network.load_state_dict(weights, strict=True)
    if weights.keys() != initial.keys() or not all(torch.isfinite(v).all() for v in weights.values()):
        raise RuntimeError("Invalid aggregate checkpoint")
    delta = sum(torch.sum((weights[key].float() - initial[key].float()) ** 2).item() for key in weights) ** 0.5
    if delta <= 0:
        raise RuntimeError("Aggregate model did not change from the pretrained weights")
    records = load_decathlon_datalist(
        str(Path(splits) / "evaluation.json"), data_list_key="validation", base_dir=str(dataset_root)
    )
    dataset = Dataset(records, transform=parser.get_parsed_content("validate#preprocessing"))
    inferer = parser.get_parsed_content("validate#inferer")
    results = []
    for item, source in zip(dataset, records):
        image = item["image"].unsqueeze(0).cuda()
        with torch.inference_mode(), torch.autocast("cuda"):
            logits = inferer(image, network)
        predicted = logits.sigmoid() > 0.5
        dice = compute_dice(predicted.cpu(), item["label"].unsqueeze(0), include_background=True, ignore_empty=False)[0]
        if not torch.isfinite(dice).all():
            raise RuntimeError("Nonfinite full-volume Dice")
        regions = predicted[0].cpu().numpy()
        # Save one mutually exclusive MSD label map, enforcing nested regions.
        segmentation = np.zeros(regions.shape[1:], dtype=np.uint8)
        segmentation[regions[0] | regions[1] | regions[2]] = 1
        segmentation[regions[0] | regions[2]] = 2
        segmentation[regions[2]] = 3
        original = nib.load(source["image"])
        if segmentation.shape != original.shape[:3]:
            raise RuntimeError("Prediction shape differs from the source volume")
        name = Path(source["image"]).name
        prediction_path = output / name.replace(".nii.gz", "_prediction.nii.gz")
        nib.save(nib.Nifti1Image(segmentation, original.affine), prediction_path)
        results.append(
            {
                "case": name,
                "dice": dict(zip(("TC", "WT", "ET"), dice.tolist())),
                "shape": list(segmentation.shape),
                "prediction": str(prediction_path),
            }
        )
    if not results:
        raise RuntimeError("No validation cases")
    mean = {region: float(np.mean([r["dice"][region] for r in results])) for region in ("TC", "WT", "ET")}
    report = {
        "checkpoint": str(checkpoint),
        "parameter_change_l2": delta,
        "mean_dice": mean,
        "mean_dice_all_regions": float(np.mean(list(mean.values()))),
        "cases": results,
        "note": "Small integration pilot; pretrained BraTS model may overlap MSD cases. Not an independent accuracy benchmark.",
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(evaluate(args.bundle_root, args.checkpoint, args.dataset_root, args.splits, args.output), indent=2)
    )
