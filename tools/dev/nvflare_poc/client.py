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

"""A real one-round NumPy client: deterministic updates, no external dataset or GPU."""

import os
from pathlib import Path

import nvflare.client as flare
from nvflare.client import FLModel

flare.init()
while flare.is_running():
    model = flare.receive()
    marker_root = os.environ.get("PF_POC_EXECUTION_MARKERS")
    if marker_root:
        Path(marker_root, f"{flare.get_job_id()}-{flare.get_site_name()}.txt").write_text("executed\n")
    flare.send(
        FLModel(params={key: value + 1.0 for key, value in model.params.items()}, meta={"NUM_STEPS_CURRENT_ROUND": 1})
    )
