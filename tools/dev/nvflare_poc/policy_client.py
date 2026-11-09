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

"""Use the installed Policy Fabric researcher client to request capabilities.

Run in its configured PDO/Django environment, with NVFlare installed alongside it.
This bridge uses the repository's Python client components, not the toy FL server
or a new capability-issuance protocol. It never fabricates credentials.
"""

import base64
import json
import os
import re
import sys
from pathlib import Path


def _load_client(webapp_dir):
    webapp = Path(webapp_dir).resolve()
    if not (webapp / "app" / "action_runners.py").is_file():
        raise ValueError("webapp_dir must point to the installed Policy Fabric client webapp")
    sys.path.insert(0, str(webapp))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django

    django.setup()
    from app import action_runners, registry_client

    # Prevent an unrelated installed package called 'app' from being used silently.
    if Path(action_runners.__file__).resolve().parent != webapp / "app":
        raise RuntimeError("A different 'app' package is already loaded; use a dedicated Policy Fabric environment")
    return action_runners, registry_client


class PolicyFabricCapabilityProvider:
    """Resolve assets, check readiness, then mint a capability for each exact digest.

    targets: [{asset_did, app_name, wallets: {role: wallet_id}, params: {...}}].
    Policy Fabric determines credential requirements, trust, and approval. A
    refusal aborts the submission; it never falls back to mock issuance.
    """

    def __init__(self, *, user_name, targets, webapp_dir=None):
        if not isinstance(user_name, str) or not user_name.strip():
            raise ValueError("user_name is required")
        if not isinstance(targets, list) or not targets:
            raise ValueError("at least one asset/application target is required")
        self.user_name = user_name
        self.targets = json.loads(json.dumps(targets))
        self.webapp_dir = webapp_dir or Path(__file__).resolve().parents[2] / "pdo_client" / "webapp"
        seen = set()
        for target in self.targets:
            for key in ("asset_did", "app_name"):
                if not isinstance(target.get(key), str) or not target[key].strip():
                    raise ValueError(f"target requires {key}")
            pair = (target["asset_did"], target["app_name"])
            if pair in seen:
                raise ValueError("duplicate asset/application target")
            seen.add(pair)
            if not isinstance(target.get("wallets"), dict) or not target["wallets"].get("Script"):
                raise ValueError("target requires wallets including the Script role")
            params = target.get("params", {})
            if not isinstance(params, dict) or "script_digest" in params:
                raise ValueError("script_digest is measured by the submission hook, never set in target params")

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def __call__(self, digests):
        if set(digests) != {target["app_name"] for target in self.targets}:
            raise ValueError("targets must cover exactly the frozen client application names")
        if any(
            not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value) for value in digests.values()
        ):
            raise ValueError("expected SHA-256 digests of the frozen application bytes")
        action_runners, registry = _load_client(self.webapp_dir)
        capabilities = {}
        for target in self.targets:
            asset_did, app_name = target["asset_did"], target["app_name"]
            asset = registry.get_asset_by_did(asset_did) or {}
            runner = action_runners.build_runner(
                user_name=self.user_name,
                asset_did=asset_did,
                metadata=asset.get("metadata") or {},
                wallets=target["wallets"],
                params={**target.get("params", {}), "script_digest": digests[app_name]},
            )
            if runner.name != "inference" or not runner.federated:
                raise ValueError(f"{asset_did}: requires an inference Guardian")
            capability = runner.mint_capability()
            if not isinstance(capability, dict) or not capability or "request_context" in capability:
                raise ValueError("Policy Fabric returned an invalid capability envelope")
            # Encode the opaque envelope for job metadata. Do not inspect or
            # rewrite the encrypted operation or use the issuer's Guardian URL
            # as a client endpoint: each site configures its own trusted endpoint.
            encoded = base64.b64encode(json.dumps(capability).encode()).decode()
            capabilities.setdefault(asset_did, {})[app_name] = encoded
        return capabilities
