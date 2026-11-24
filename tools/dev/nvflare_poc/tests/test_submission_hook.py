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

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from demo import make_recipe
from job_hash import hash_bytes
from nvflare.recipe.session_mgr import SessionManager
from pf_nvflare import CAPABILITIES, PAYLOAD_FILE
from prepare_job import PolicyJobPreparation, prepare_job
from submission_hook import PolicyProdEnv


@pytest.fixture
def submission(monkeypatch):
    state = SimpleNamespace(paths=[], metadata=[], payloads=[], connections=0, closes=0, failure=None)

    def submit(path):
        job_dir = Path(path)
        state.paths.append(job_dir)
        state.metadata.append(json.loads((job_dir / "meta.json").read_text()))
        state.payloads.append({p.parent.name: p.read_bytes() for p in job_dir.rglob(PAYLOAD_FILE)})
        if state.failure:
            raise state.failure
        return f"job-{len(state.paths)}"

    def close():
        state.closes += 1

    def connect(manager):
        state.connections += 1
        return SimpleNamespace(submit_job=submit, close=close, get_job_status=lambda _: "RUNNING")

    monkeypatch.setattr(SessionManager, "_get_session", connect)
    return state


def test_recipe_execute_prepares_final_export_before_submit(tmp_path, submission, monkeypatch):
    order = []
    paths = []

    def prepare(path):
        assert (path / "app_site_a" / "config" / "config_fed_client.json").is_file()
        assert (path / "app_site_b" / "custom" / "client.py").is_file()
        for site in ("site_a", "site_b"):
            config = (path / f"app_{site}" / "config/config_fed_client.json").read_text()
            assert "custom/client.py" in config
            assert str(Path(__file__).resolve().parents[1] / "client.py") not in config
        paths.append(path)
        order.append("hook")
        prepare_job(path, ("policy-a", "policy-b"))

    def scan(*args, **kwargs):
        order.append("scan")

    monkeypatch.setattr("submission_hook.warn_on_potential_secrets_in_job_dir", scan)
    env = PolicyProdEnv(str(tmp_path), before_submit=prepare)
    execution = make_recipe("native-recipe").execute(env)
    assert execution.get_job_id() == "job-1"
    assert order == ["hook", "scan"]
    assert submission.paths == paths
    assert submission.metadata[0]["custom_props"][CAPABILITIES].keys() == {"policy-a", "policy-b"}
    assert submission.payloads[0].keys() == {"app_site_a", "app_site_b"}
    assert submission.closes == 1
    assert not paths[0].exists()
    assert execution.get_status() == "RUNNING"  # inherited ProdEnv/Run behavior
    assert submission.closes == 2


def test_hook_exception_prevents_connection_and_cleans_export(tmp_path, submission):
    paths = []

    def reject(path):
        paths.append(path)
        raise ValueError("policy approval unavailable")

    env = PolicyProdEnv(str(tmp_path), before_submit=reject)
    with pytest.raises(RuntimeError, match="policy approval unavailable") as error:
        make_recipe("rejected").execute(env)
    assert isinstance(error.value.__cause__, ValueError)
    assert submission.connections == 0
    assert not paths[0].exists()


@pytest.mark.parametrize("result", [False, True, "another/path"])
def test_hook_return_value_cannot_silently_authorize(tmp_path, submission, result):
    env = PolicyProdEnv(str(tmp_path), before_submit=lambda _: result)
    with pytest.raises(RuntimeError, match="must return None"):
        make_recipe("bad-return").execute(env)
    assert submission.connections == 0


def test_no_hook_keeps_original_submission_path(tmp_path, submission):
    env = PolicyProdEnv(str(tmp_path))
    assert type(env._get_session_manager()) is SessionManager
    assert make_recipe("ordinary").execute(env).get_job_id() == "job-1"
    assert submission.payloads == [{}]
    assert submission.closes == 1
    assert not submission.paths[0].exists()


def test_submit_failure_closes_session_without_retry(tmp_path, submission):
    calls = []
    submission.failure = OSError("server unavailable")

    def prepare(path):
        calls.append(path)

    with pytest.raises(RuntimeError, match="server unavailable"):
        make_recipe("submit-failure").execute(PolicyProdEnv(str(tmp_path), before_submit=prepare))
    assert len(calls) == submission.connections == submission.closes == 1
    assert not calls[0].exists()


def test_repeated_submissions_get_fresh_exports(tmp_path, submission):
    paths = []

    def prepare(path):
        paths.append(path)
        assert not list(path.rglob(PAYLOAD_FILE))
        prepare_job(path, ("policy-a",))

    env = PolicyProdEnv(str(tmp_path), before_submit=prepare)
    recipe = make_recipe("repeated")
    recipe.execute(env)
    recipe.execute(env)
    assert len(set(paths)) == 2
    assert submission.connections == submission.closes == 2
    assert all(not path.exists() for path in paths)


def test_export_only_does_not_run_submission_hook(tmp_path, monkeypatch, submission):
    env = PolicyProdEnv(str(tmp_path), before_submit=lambda _: pytest.fail("export is not submission"))
    export = tmp_path / "export"
    # NVFlare consumes these flags at import time; emulate that recorded choice.
    monkeypatch.setattr("nvflare.recipe.spec._peek_recipe_args", lambda: (True, str(export)))
    with pytest.raises(SystemExit) as exit_info:
        make_recipe("export-only").execute(env)
    assert exit_info.value.code == 0
    assert (export / "export-only" / "meta.json").is_file()
    assert submission.connections == 0


def test_async_hook_is_rejected(tmp_path):
    async def prepare(path):
        pass

    with pytest.raises(TypeError, match="synchronous"):
        PolicyProdEnv(str(tmp_path), before_submit=prepare)


def test_policy_provider_receives_hash_of_signed_frozen_bytes(tmp_path, submission, monkeypatch):
    signs = []
    digests_seen = {}
    monkeypatch.setattr("prepare_job.load_private_key_file", lambda _: "test-key")

    def sign(folder, key, cert):
        folder = Path(folder)
        signs.append(folder)
        if len(signs) == 1:
            # Stand in for signing artifacts that must be inside the measured ZIP.
            for app in (folder / "app_site_a", folder / "app_site_b"):
                (app / "signature-test.txt").write_text("signed application")
        else:
            assert json.loads((folder / "meta.json").read_text())["custom_props"][CAPABILITIES]

    def provider(digests):
        assert len(signs) == 1
        for app, digest in digests.items():
            payload = signs[0] / app / PAYLOAD_FILE
            assert hash_bytes(payload.read_bytes()) == digest
            with zipfile.ZipFile(payload) as archive:
                assert archive.read("signature-test.txt") == b"signed application"
        digests_seen.update(digests)
        return {"policy-a": {app: "opaque-capability" for app in digests}}

    monkeypatch.setattr("prepare_job.sign_folders", sign)
    prepare = PolicyJobPreparation(tmp_path, capability_provider=provider)
    make_recipe("provider").execute(PolicyProdEnv(str(tmp_path), before_submit=prepare))
    assert len(signs) == 2
    assert prepare.digests == digests_seen
    for app, payload in submission.payloads[0].items():
        assert hash_bytes(payload) == digests_seen[app]


def test_capability_provider_failure_prevents_submission(tmp_path, submission, monkeypatch):
    monkeypatch.setattr("prepare_job.load_private_key_file", lambda _: "test-key")
    monkeypatch.setattr("prepare_job.sign_folders", lambda *args: None)

    def reject(digests):
        raise ValueError("capability denied")

    prepare = PolicyJobPreparation(tmp_path, capability_provider=reject)
    with pytest.raises(RuntimeError, match="capability denied"):
        make_recipe("provider-denial").execute(PolicyProdEnv(str(tmp_path), before_submit=prepare))
    assert submission.connections == 0
    assert prepare.digests == {}


def test_delayed_approval_reuses_exact_frozen_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr("prepare_job.load_private_key_file", lambda _: "test-key")
    monkeypatch.setattr("prepare_job.sign_folders", lambda *args: None)
    recipe = make_recipe("delayed-approval")
    recipe.export(str(tmp_path / "exports"))
    job_dir = tmp_path / "exports" / "delayed-approval"
    approved = {}

    def provider(digests):
        assert digests == approved
        return {"did:pdo:asset": {app: "opaque" for app in digests}}

    preparation = PolicyJobPreparation(tmp_path, capability_provider=provider)
    approved.update(preparation.freeze(job_dir))
    original = {app: (job_dir / app / PAYLOAD_FILE).read_bytes() for app in approved}
    assert json.loads((job_dir / "meta.json").read_text())["custom_props"][CAPABILITIES] == {}
    preparation.authorize(job_dir)
    assert preparation.digests == approved
    assert {app: (job_dir / app / PAYLOAD_FILE).read_bytes() for app in approved} == original
