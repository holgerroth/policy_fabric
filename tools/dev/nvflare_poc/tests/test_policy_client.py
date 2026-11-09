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

"""Client/operation contract tests. PDO cryptography and the ledger are not simulated."""

import base64
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from job_hash import hash_bytes
from policy_client import PolicyFabricCapabilityProvider

TOOLS = Path(__file__).resolve().parents[3]
DIGEST = hash_bytes(b"signed application ZIP")
ENVELOPE = {"minted_identity": "test-issuer", "operation": {"encrypted_message": "opaque-ciphertext"}}


@pytest.fixture
def upstream_runner(monkeypatch):
    # Execute the actual upstream runner; replace only its external services.
    prefix = "_policy_client_contract"
    package = ModuleType(prefix)
    package.__path__ = [str(TOOLS / "pdo_client" / "webapp" / "app")]
    monkeypatch.setitem(sys.modules, prefix, package)
    calls = []

    def issue(**kwargs):
        calls.append(kwargs)
        return ENVELOPE, {}

    for name, attrs in {
        "pdo_runner": {"create_capability": issue},
        "readiness": {
            "assess": lambda *a: {"trusted": [], "needs_project": False, "rows": []},
            "obtain": lambda *a: [],
        },
        "checklist": {"ready": lambda rows: True},
        "registry_client": {},
        "session_keys": {},
        "did_utils": {"parse_did": lambda value: (value.removeprefix("did:pdo:"), None)},
        "views": {},
        "views._streaming": {"SkipStep": type("SkipStep", (Exception,), {})},
    }.items():
        module = ModuleType(prefix + "." + name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location(
        prefix + ".action_runners", package.__path__[0] + "/action_runners.py"
    )
    runners = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runners)
    registry = SimpleNamespace(
        get_asset_by_did=lambda _: {
            "metadata": {
                "guardian_type": "inference",
                "guardian_url": "issuer-side-address",
                "guardian_port": "7900",
                "policy_contract": "did:pdo:token-contract",
            }
        }
    )
    monkeypatch.setattr("policy_client._load_client", lambda _: (runners, registry))
    return runners, registry, calls


def provider(**changes):
    target = {
        "asset_did": "did:pdo:data-asset",
        "app_name": "app",
        "wallets": {"Script": "archive-identity", "User": "researcher-wallet"},
        "params": {"project": "approved-project"},
    }
    target.update(changes)
    return PolicyFabricCapabilityProvider(user_name="researcher", targets=[target])


def test_actual_upstream_runner_receives_archive_digest_and_roles(upstream_runner):
    _, _, calls = upstream_runner
    bundle = provider()({"app": DIGEST})
    assert json.loads(base64.b64decode(bundle["did:pdo:data-asset"]["app"])) == ENVELOPE
    assert calls == [
        {
            "wallet_ids": {"Script": "archive-identity", "User": "researcher-wallet"},
            "token_id": "token-contract",
            "user_name": "researcher",
            "request": {"project": "approved-project", "script_digest": DIGEST},
        }
    ]


def test_policy_refusal_propagates_without_mock_fallback(upstream_runner, monkeypatch):
    runners, _, _ = upstream_runner

    def refuse(**kwargs):
        raise RuntimeError("policy denied: missing trusted ScriptHashCredential")

    monkeypatch.setattr(runners.pdo_runner, "create_capability", refuse)
    with pytest.raises(RuntimeError, match="policy denied"):
        provider()({"app": DIGEST})


def test_non_inference_asset_is_rejected(upstream_runner):
    _, registry, _ = upstream_runner
    registry.get_asset_by_did = lambda _: {
        "metadata": {"guardian_type": "public", "guardian_url": "x", "guardian_port": "1"}
    }
    with pytest.raises(ValueError, match="requires an inference"):
        provider()({"app": DIGEST})


@pytest.mark.parametrize("digests", [{}, {"another-app": DIGEST}, {"app": "sha256:not-a-hash"}])
def test_missing_target_or_invalid_measurement_fails_before_client_load(digests, monkeypatch):
    monkeypatch.setattr("policy_client._load_client", lambda _: pytest.fail("must validate locally first"))
    with pytest.raises(ValueError):
        provider()(digests)


def test_config_cannot_override_measured_digest():
    with pytest.raises(ValueError, match="measured"):
        provider(params={"script_digest": "forged"})


def test_required_script_identity_is_not_fabricated():
    with pytest.raises(ValueError, match="Script"):
        provider(wallets={"User": "researcher-wallet"})


def test_missing_readiness_stops_issuance(upstream_runner, monkeypatch):
    runners, _, calls = upstream_runner
    monkeypatch.setattr(runners.checklist, "ready", lambda _: False)
    monkeypatch.setattr(runners.readiness, "describe_blockers", lambda _: "trusted approval missing", raising=False)
    with pytest.raises(ValueError, match="Not ready"):
        provider()({"app": DIGEST})
    assert calls == []
