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

import base64
import importlib.util
import io
import json
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from job_hash import hash_bytes, hash_file
from nvflare.apis.client import Client
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from pf_nvflare import (
    CAPABILITIES,
    PAYLOAD_FILE,
    ROUTES,
    FrozenAppJobManager,
    GuardianClient,
    PolicyAppDeployer,
    PolicyError,
    PolicyRegistrationRouter,
    PolicySecurityHandler,
)
from prepare_job import make_mock_capability

POLICY = "did:pdo:test-policy"


def encode_envelope(envelope):
    return base64.b64encode(json.dumps(envelope).encode()).decode()


def test_pdo_wire_format_and_actual_upstream_operation(serve, monkeypatch):
    # The operation is real upstream code. The outer PDO crypto layer is not
    # exercised here: the test server supplies already-authenticated parameters.
    utility = ModuleType("pdo.contracts.guardian.common.utility")
    utility.ValidateJSON = lambda value, schema: isinstance(value, dict) and all(
        isinstance(value.get(key), str) for key in schema["required"]
    )
    monkeypatch.setitem(sys.modules, utility.__name__, utility)
    path = Path(__file__).resolve().parents[3] / "guardians/inference/guardian_core/inference.py"
    spec = importlib.util.spec_from_file_location("upstream_inference_operation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    operation = module.InferenceOperation({})
    digest = hash_bytes(b"frozen application")
    envelope = {"minted_identity": "test", "operation": {"encrypted_message": "opaque"}}
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            decision = operation({"script_digest": digest}, body["request_context"])
            self.send_response(200 if decision else 422)
            self.end_headers()
            self.wfile.write(json.dumps(decision).encode())

    client = GuardianClient(serve(Handler))
    client.verify(encode_envelope(envelope), digest)
    assert requests == [{**envelope, "request_context": {"calculated_script_digest": digest}}]
    with pytest.raises(PolicyError, match="422"):
        client.verify(encode_envelope(envelope), hash_bytes(b"tampered"))


@pytest.mark.parametrize("mode,marker", [("pdo", {"mock": True}), ("mock", {})])
def test_guardian_modes_cannot_be_silently_interchanged(serve, mode, marker):
    digest = hash_bytes(b"x")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps({"allowed": True, "script_digest": digest, "method_name": "do_inference", **marker}).encode()
            )

    with pytest.raises(PolicyError, match="matching"):
        GuardianClient(serve(Handler), mode=mode).verify(encode_envelope({"operation": {}}), digest)


@pytest.mark.parametrize("envelope", [[], {}, {"request_context": {"calculated_script_digest": "forged"}}])
def test_invalid_pdo_envelope_is_rejected_locally(envelope):
    with pytest.raises(PolicyError, match="envelope"):
        GuardianClient("http://127.0.0.1:1").verify(encode_envelope(envelope), hash_bytes(b"x"))


@pytest.fixture
def ctx():
    result = FLContext()
    result.set_prop(ReservedKey.IDENTITY_NAME, "site_a")
    result.set_prop(FLContextKey.CLIENT_CONFIG, {"site_config": {"asset_did": POLICY}})
    return result


def metadata(capability="opaque", policy=POLICY):
    return {
        "custom_props": {ROUTES: {"site_a": {"asset_did": policy, "app_name": "app", "capability_b64": capability}}}
    }


@pytest.fixture
def serve():
    running = []

    def start(handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        running.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}"

    yield start
    for server, thread in running:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def guardian(serve):
    path = Path(__file__).resolve().parents[2] / "mock_inference_guardian" / "server.py"
    spec = importlib.util.spec_from_file_location("upstream_mock_guardian", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return GuardianClient(serve(module.MockGuardianHandler), mode="mock")


def test_generic_hash(tmp_path):
    path = tmp_path / "any-framework.job"
    path.write_bytes(b"abc")
    assert hash_bytes(b"abc") == "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert hash_file(path) == hash_bytes(b"abc")
    with pytest.raises(TypeError):
        hash_bytes("abc")


def test_upstream_guardian_accepts_and_denies(guardian):
    digest = hash_bytes(b"application")
    capability = make_mock_capability(digest)
    guardian.verify(capability, digest)
    with pytest.raises(PolicyError, match="422"):
        guardian.verify(capability, hash_bytes(b"tampered"))
    with pytest.raises(PolicyError, match="400"):
        guardian.verify("not-base64", digest)


@pytest.mark.parametrize("body", [{"allowed": False}, {"allowed": "true"}, {}, [], "asset-data"])
def test_ambiguous_responses_fail_closed(serve, body):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

    with pytest.raises(PolicyError):
        GuardianClient(serve(Handler), mode="mock").verify("opaque", hash_bytes(b"x"))


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://127.0.0.1",
        "http://localhost@evil.test",
        "http://localhost/path",
        "http://localhost?x=1",
    ],
)
def test_guardian_endpoint_is_site_local(url):
    with pytest.raises(ValueError):
        GuardianClient(url)


def test_routing_uses_registration_not_site_name(ctx):
    client = Client("site_a", "token")
    client.set_site_config({"asset_did": POLICY})
    engine = SimpleNamespace(get_client_from_name=lambda name: client, fire_event=lambda *_: None)
    ctx.set_prop(ReservedKey.ENGINE, engine)
    meta = {
        "deploy_map": {"app": ["site_a"]},
        "custom_props": {CAPABILITIES: {POLICY: {"app": "opaque"}}, ROUTES: {"site_a": "forged"}},
    }
    ctx.set_prop(FLContextKey.JOB_META, meta)
    ctx.set_prop(FLContextKey.CLIENT_RESOURCE_SPECS, {"site_a": {}})
    PolicyRegistrationRouter().handle_event(EventType.BEFORE_CHECK_CLIENT_RESOURCES, ctx)
    assert meta["custom_props"][ROUTES]["site_a"]["capability_b64"] == "opaque"
    assert meta["custom_props"][ROUTES]["site_a"]["asset_did"] == POLICY


@pytest.mark.parametrize("meta", [{}, metadata(None), metadata(policy="somebody-else"), {"custom_props": []}])
def test_early_authorization_denies_missing_or_wrong_route(ctx, meta):
    ctx.set_prop(FLContextKey.COMMAND_NAME, "check_resources")
    ctx.set_prop(FLContextKey.SECURITY_ITEMS, {FLContextKey.JOB_META: meta})
    PolicySecurityHandler().handle_event(EventType.AUTHORIZE_COMMAND_CHECK, ctx)
    assert ctx.get_prop(FLContextKey.AUTHORIZATION_RESULT) is False


def test_allow_does_not_override_existing_deny(ctx):
    ctx.set_prop(FLContextKey.AUTHORIZATION_RESULT, False)
    ctx.set_prop(FLContextKey.COMMAND_NAME, "check_resources")
    ctx.set_prop(FLContextKey.SECURITY_ITEMS, {FLContextKey.JOB_META: metadata()})
    PolicySecurityHandler().handle_event(EventType.AUTHORIZE_COMMAND_CHECK, ctx)
    assert ctx.get_prop(FLContextKey.AUTHORIZATION_RESULT) is False


def test_actual_bytes_checked_before_unpacking(ctx, guardian, monkeypatch):
    from nvflare.private.fed.utils.app_deployer import AppDeployer

    deployed = []
    monkeypatch.setattr(AppDeployer, "deploy", lambda *args: deployed.append(args[5]))
    plugin = PolicyAppDeployer("http://localhost:1")
    plugin.guardian = guardian
    payload = b"approved archive bytes"
    meta = metadata(make_mock_capability(hash_bytes(payload)))
    assert plugin.deploy(None, "job", meta, "app", payload, ctx) is None
    assert deployed == [payload]
    assert "denied" in plugin.deploy(None, "job", meta, "app", b"tampered", ctx)
    assert deployed == [payload]


def test_failed_guardian_never_calls_default_deployer(ctx, monkeypatch):
    from nvflare.private.fed.utils.app_deployer import AppDeployer

    monkeypatch.setattr(AppDeployer, "deploy", lambda *args: pytest.fail("must not unpack"))
    plugin = PolicyAppDeployer("http://127.0.0.1:1", timeout=0.1)
    assert "denied" in plugin.deploy(None, "job", metadata(), "app", b"x", ctx)


def test_frozen_archive_survives_nvflare_repacking(tmp_path, monkeypatch):
    from nvflare.apis.impl.job_def_manager import SimpleJobDefManager

    buffer = io.BytesIO()
    frozen = b"exact approved serialized bytes"
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(PAYLOAD_FILE, frozen)
        archive.writestr("config/config_fed_client.json", "{}")
    monkeypatch.setattr(SimpleJobDefManager, "get_app", lambda *args: buffer.getvalue())
    manager = FrozenAppJobManager(str(tmp_path / "store"))
    job = SimpleNamespace(get_deployment=lambda: {"app": ["site_a"]})
    assert manager.get_app(job, "app", None) == frozen


@pytest.mark.parametrize(
    "status,raw", [(200, b"not JSON"), (200, b"x" * 65537), (503, b"unavailable"), (302, b"redirect")]
)
def test_http_failures_and_redirects_are_denials(serve, status, raw):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(status)
            if status == 302:
                self.send_header("Location", "/redirected")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            pytest.fail("redirect must not be followed")

    with pytest.raises(PolicyError):
        GuardianClient(serve(Handler), mode="mock").verify("opaque", hash_bytes(b"x"))


def test_guardian_timeout_denies(serve):
    class SlowHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            time.sleep(0.2)
            self.send_response(200)
            self.end_headers()

    with pytest.raises(PolicyError, match="unavailable"):
        GuardianClient(serve(SlowHandler), timeout=0.02, mode="mock").verify("opaque", hash_bytes(b"x"))


def test_allowed_response_must_bind_measured_digest(serve):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "allowed": True,
                        "mock": True,
                        "method_name": "do_inference",
                        "script_digest": hash_bytes(b"different"),
                    }
                ).encode()
            )

    with pytest.raises(PolicyError, match="matching"):
        GuardianClient(serve(Handler), mode="mock").verify("opaque", hash_bytes(b"x"))


def test_existing_deployer_denial_is_preserved(ctx, guardian, monkeypatch):
    from nvflare.private.fed.utils.app_deployer import AppDeployer

    monkeypatch.setattr(AppDeployer, "deploy", lambda *args: "privacy scope not allowed")
    plugin = PolicyAppDeployer("http://localhost:1")
    plugin.guardian = guardian
    meta = metadata(make_mock_capability(hash_bytes(b"x")))
    assert plugin.deploy(None, "job", meta, "app", b"x", ctx) == "privacy scope not allowed"


def test_app_route_mismatch_never_contacts_guardian(ctx):
    plugin = PolicyAppDeployer("http://localhost:1")
    plugin.guardian = SimpleNamespace(verify=lambda *args: pytest.fail("must not contact guardian"))
    assert "different application" in plugin.deploy(None, "job", metadata(), "another-app", b"x", ctx)


def test_missing_frozen_payload_fails_closed(tmp_path, monkeypatch):
    from nvflare.apis.impl.job_def_manager import SimpleJobDefManager

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("custom/client.py", "print('unapproved')")
    monkeypatch.setattr(SimpleJobDefManager, "get_app", lambda *args: buffer.getvalue())
    manager = FrozenAppJobManager(str(tmp_path / "store"))
    job = SimpleNamespace(get_deployment=lambda: {"app": ["site_a"]})
    with pytest.raises(PolicyError, match="frozen"):
        manager.get_app(job, "app", None)
