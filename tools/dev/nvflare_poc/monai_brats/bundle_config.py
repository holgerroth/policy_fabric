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

"""Small data/runtime overrides; architecture, loss and augmentation come from the bundle."""

import json
from pathlib import Path


def overrides(bundle_root, dataset_root, datalist, output_root, epochs):
    config = json.loads((Path(bundle_root) / "configs/train.json").read_text())
    preprocessing = [
        {"_target_": "LoadImaged", "keys": ["image", "label"], "image_only": False},
        {"_target_": "EnsureChannelFirstd", "keys": "image", "channel_dim": -1},
        {"_target_": "msd_transforms.MsdBraTSChannelsd"},
        {"_target_": "NormalizeIntensityd", "keys": "image", "nonzero": True, "channel_wise": True},
    ]
    return {
        "dataset_dir": str(Path(dataset_root).resolve()),
        "data_list_file_path": str(Path(datalist).resolve()),
        "output_dir": str(Path(output_root).resolve()),
        "ckpt_dir": str(Path(output_root).resolve() / "models"),
        "epochs": epochs,
        "train#preprocessing_transforms": preprocessing,
        "train#dataloader#num_workers": 0,
        "validate#dataloader#num_workers": 0,
        # Evaluation happens explicitly in the FL loop; avoid a second pass
        # from the trainer and avoid overwriting the downloaded bundle checkpoint.
        "train#handlers": [h for h in config["train"]["handlers"] if h["_target_"] != "ValidationHandler"],
        "validate#handlers": [h for h in config["validate"]["handlers"] if h["_target_"] != "CheckpointSaver"],
    }
