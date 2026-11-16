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

"""Adapt the official MSD MRI layout to the unmodified BraTS bundle network."""

import numpy as np
import torch
from monai.transforms import MapTransform


class MsdBraTSChannelsd(MapTransform):
    """MSD FLAIR/T1/T1gd/T2 -> bundle T1c/T1/T2/FLAIR; labels -> TC/WT/ET.

    MSD dataset.json declares 1=edema, 2=non-enhancing tumor, 3=enhancing
    tumor. The bundle's stock transform expects BraTS18 labels 1/2/4.
    The preparation step checks the dataset metadata before creating splits.
    """

    def __init__(self, keys=("image", "label")):
        super().__init__(keys)

    def __call__(self, data):
        result = dict(data)
        image = result["image"]
        label = result["label"]
        if image.ndim != 4 or image.shape[0] != 4 or label.ndim != 3 or image.shape[1:] != label.shape:
            raise ValueError("Expected a four-channel MSD image and matching 3D label")
        values = torch.unique(label).cpu().tolist() if isinstance(label, torch.Tensor) else np.unique(label).tolist()
        if not set(values) <= {0, 1, 2, 3}:
            raise ValueError(f"Unexpected MSD label values: {values}")
        result["image"] = image[[2, 1, 3, 0]]
        regions = [(label == 2) | (label == 3), label > 0, label == 3]
        result["label"] = (
            torch.stack(regions).float() if isinstance(label, torch.Tensor) else np.stack(regions).astype(np.float32)
        )
        return result
