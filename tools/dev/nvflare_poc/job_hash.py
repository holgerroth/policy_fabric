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

"""Framework-independent hashing of an already serialized job. No manifest or adapters."""

import argparse
import hashlib
from pathlib import Path


def hash_bytes(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise TypeError("serialized payload must be bytes")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def hash_file(path: str | Path) -> str:
    with open(path, "rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    print(hash_file(parser.parse_args().artifact))
