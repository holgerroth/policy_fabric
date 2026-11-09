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

"""Freeze exported client applications and attach opaque capabilities outside their hashes."""

import argparse
import base64
import io
import json
import secrets
import zipfile
from pathlib import Path

from job_hash import hash_bytes
from nvflare.lighter.utils import load_private_key_file, sign_folders
from pf_nvflare import CAPABILITIES, PAYLOAD_FILE


def make_mock_capability(digest: str) -> str:
    """Portable equivalent of the upstream mock generator. NEVER a real authorization."""

    def encode(value):
        return base64.b64encode(json.dumps(value).encode()).decode()

    operation = {
        "nonce": secrets.token_hex(16),
        "request_identifier": secrets.token_hex(8),
        "method_name": "do_inference",
        "parameters": {"script_digest": digest},
    }
    return encode(
        {
            "minted_identity": "MOCK-NOT-A-TRUSTED-ISSUER",
            "operation": {
                "encrypted_session_key": "MOCK",
                "session_key_iv": "MOCK",
                "encrypted_message": encode(operation),
            },
        }
    )


def freeze_app(app_dir: Path) -> bytes:
    """Serialize once. The generic hashing function knows nothing about this layout."""
    target = app_dir / PAYLOAD_FILE
    if target.exists():
        raise ValueError(f"already frozen: {target}; export a fresh job to change it")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(app_dir.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"symlinks are not supported in the PoC: {path}")
            if path.is_file():
                archive.write(path, path.relative_to(app_dir).as_posix())
    payload = buffer.getvalue()
    target.write_bytes(payload)
    return payload


def prepare_job(job_dir: Path, asset_dids=(), capabilities=None):
    meta_path = job_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    approved = {} if capabilities is None else capabilities
    digests = {}
    for app, sites in meta["deploy_map"].items():
        if sites == ["server"]:
            continue
        if not isinstance(sites, list) or any(not isinstance(site, str) for site in sites):
            raise ValueError("PoC requires a flat deployment map")
        digest = hash_bytes(freeze_app(job_dir / app))
        digests[app] = digest
        for asset_did in asset_dids:
            approved.setdefault(asset_did, {})[app] = make_mock_capability(digest)
    meta.setdefault("custom_props", {})[CAPABILITIES] = approved
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return digests


class PolicyJobPreparation:
    """Submission callback: sign, freeze, attach capabilities, then sign the job.

    Mock issuance must be explicitly selected. Alternatively, a caller-provided
    capability provider receives {app_name: digest} after freezing and returns
    {asset_did: {app_name: opaque_capability}}. ``freeze`` and ``authorize`` can
    also be called separately when exact archive bytes need external approval.
    ``digests`` records the latest successful authorization for local reporting.
    """

    def __init__(self, startup_kit_location, *, mock_asset_dids=(), capability_provider=None):
        self.mock_asset_dids = tuple(mock_asset_dids)
        if bool(self.mock_asset_dids) == (capability_provider is not None):
            raise ValueError("choose either nonempty mock_asset_dids or a capability_provider")
        if capability_provider is not None and not callable(capability_provider):
            raise TypeError("capability_provider must be callable")
        self.startup = Path(startup_kit_location) / "startup"
        self.capability_provider = capability_provider
        self.digests = {}

    def __call__(self, job_dir: Path) -> None:
        self.freeze(job_dir)
        self.authorize(job_dir)

    def freeze(self, job_dir: Path):
        """Sign and freeze once; obtain approval for these exact bytes afterwards."""
        self.digests = {}
        signing_key = load_private_key_file(str(self.startup / "client.key"))
        certificate = str(self.startup / "client.crt")
        sign_folders(str(job_dir), signing_key, certificate)
        return prepare_job(job_dir)

    def authorize(self, job_dir: Path) -> None:
        """Measure existing frozen archives, obtain capabilities, sign the outer job."""
        self.digests = {}
        meta_path = job_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        digests = {
            app: hash_bytes((job_dir / app / PAYLOAD_FILE).read_bytes())
            for app, sites in meta["deploy_map"].items()
            if sites != ["server"]
        }
        if self.capability_provider is not None:
            capabilities = self.capability_provider(digests.copy())
            if not isinstance(capabilities, dict):
                raise TypeError("capability_provider must return a capability bundle dictionary")
        else:
            capabilities = {
                asset: {app: make_mock_capability(digest) for app, digest in digests.items()}
                for asset in self.mock_asset_dids
            }
        meta.setdefault("custom_props", {})[CAPABILITIES] = capabilities
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        signing_key = load_private_key_file(str(self.startup / "client.key"))
        certificate = str(self.startup / "client.crt")
        sign_folders(str(job_dir), signing_key, certificate)
        self.digests = digests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--startup-kit", type=Path, required=True, help="NVFlare administrator startup kit")
    parser.add_argument(
        "--authorize-only", action="store_true", help="Use existing frozen archives without re-exporting"
    )
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument(
        "--freeze-only", action="store_true", help="Print digests for approval; do not request capabilities"
    )
    choice.add_argument("--mock-asset-did", action="append", help="Explicitly mint INSECURE mock capabilities")
    choice.add_argument("--policy-client", type=Path, help="JSON configuration for the installed Policy Fabric client")
    args = parser.parse_args()
    if args.freeze_only and args.authorize_only:
        parser.error("--freeze-only and --authorize-only are mutually exclusive")
    if args.policy_client:
        from policy_client import PolicyFabricCapabilityProvider

        provider = PolicyFabricCapabilityProvider.from_file(args.policy_client)
    else:
        provider = (lambda _: {}) if args.freeze_only else None
    preparation = PolicyJobPreparation(
        args.startup_kit, mock_asset_dids=args.mock_asset_did or (), capability_provider=provider
    )
    if args.freeze_only:
        digests = preparation.freeze(args.job)
    else:
        if args.authorize_only:
            preparation.authorize(args.job)
        else:
            preparation(args.job)
        digests = preparation.digests
    print(json.dumps(digests, indent=2))
