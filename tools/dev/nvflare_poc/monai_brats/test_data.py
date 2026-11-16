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

"""Check the clinically meaningful channel/region conversion and patient separation."""

import json

import numpy as np
import pytest
import torch
from monai.data import MetaTensor
from msd_transforms import MsdBraTSChannelsd
from prepare_data import prepare


@pytest.mark.parametrize("kind", [np.asarray, torch.as_tensor, MetaTensor])
def test_msd_channels_and_regions(kind):
    image = kind(np.broadcast_to(np.arange(4)[:, None, None, None], (4, 2, 2, 1)).copy())
    label = kind(np.arange(4).reshape(2, 2, 1))
    result = MsdBraTSChannelsd()({"image": image, "label": label})
    np.testing.assert_array_equal(result["image"][:, 0, 0, 0], [2, 1, 3, 0])
    np.testing.assert_array_equal(result["label"].reshape(3, 4), [[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]])
    np.testing.assert_array_equal(label.reshape(-1), [0, 1, 2, 3])


def test_reject_bra_ts18_labels_and_wrong_dimensions():
    transform = MsdBraTSChannelsd()
    with pytest.raises(ValueError, match="label values"):
        transform({"image": np.zeros((4, 2, 2, 1)), "label": np.full((2, 2, 1), 4)})
    with pytest.raises(ValueError, match="four-channel"):
        transform({"image": np.zeros((3, 2, 2, 1)), "label": np.zeros((2, 2, 1))})


def test_patient_splits_are_disjoint_and_repeatable(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    records = []
    for index in range(20):
        record = {"image": f"image_{index}.nii.gz", "label": f"label_{index}.nii.gz"}
        for filename in record.values():
            (data / filename).touch()
        records.append(record)
    metadata = {
        "name": "BRATS",
        "modality": {"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"},
        "labels": {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"},
        "training": records,
    }
    (data / "dataset.json").write_text(json.dumps(metadata))
    a = prepare(data, tmp_path / "a")
    b = prepare(data, tmp_path / "b")
    assert a == b
    groups = [set(group["cases"]) for site in a["sites"].values() for group in site.values()]
    assert [len(group) for group in groups] == [4, 2, 4, 2]
    assert len(set.union(*groups)) == 12
    heldout = json.loads((tmp_path / "a/evaluation.json").read_text())["validation"]
    assert {record["image"] for record in heldout} == groups[1] | groups[3]
    with pytest.raises(ValueError, match="Not enough cases"):
        prepare(data, tmp_path / "bad", train_per_site=20)
