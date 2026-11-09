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

"""Site-installed Policy Fabric plugins. Never install these from submitted job code."""

import base64
import binascii
import io
import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from job_hash import hash_bytes
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.impl.job_def_manager import SimpleJobDefManager
from nvflare.apis.job_def import JobMetaKey
from nvflare.private.fed.utils.app_deployer import AppDeployer

PAYLOAD_FILE = "policy_payload.zip"
CAPABILITIES = "policy_fabric_capabilities"
ROUTES = "policy_fabric_routes"
MAX_RESPONSE_BYTES = 65536


class PolicyError(ValueError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GuardianClient:
    """Decision client for the upstream inference Guardian; mock mode is explicit."""

    def __init__(self, url: str, timeout: float = 5.0, mode: str = "pdo"):
        if mode not in ("pdo", "mock"):
            raise ValueError("Guardian mode must be 'pdo' or 'mock'")
        self.mode = mode
        parsed = urllib.parse.urlsplit(url)
        try:
            local = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            local = False
        if (
            not local
            or parsed.scheme != "http"
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError("PoC Guardian must be a site-configured loopback HTTP origin")
        if not 0 < timeout <= 30:
            raise ValueError("Guardian timeout must be in (0, 30] seconds")
        self.url = url.rstrip("/") + "/process_capability"
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def verify(self, capability_b64: str, measured_digest: str) -> None:
        if self.mode == "mock":
            body = {"capability_b64": capability_b64}
        else:
            # Unwrap transport encoding only. PDO retains responsibility for
            # decrypting/authenticating the operation inside this envelope.
            try:
                if len(capability_b64) > MAX_RESPONSE_BYTES:
                    raise ValueError("capability too large")
                body = json.loads(base64.b64decode(capability_b64, validate=True))
                if not isinstance(body, dict) or not body or "request_context" in body:
                    raise ValueError("invalid capability envelope")
            except (ValueError, TypeError, binascii.Error) as exc:
                raise PolicyError("Invalid PDO capability envelope") from exc
        body["request_context"] = {"calculated_script_digest": measured_digest}
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise PolicyError(f"Guardian returned HTTP {response.status}")
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise PolicyError("Guardian response is too large")
            decision = json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise PolicyError(f"Guardian denied request: HTTP {exc.code}") from exc
        except (OSError, ValueError) as exc:
            raise PolicyError(f"Guardian unavailable or invalid response ({type(exc).__name__})") from exc
        if not isinstance(decision, dict) or not (
            decision.get("allowed") is True
            and decision.get("script_digest") == measured_digest
            and decision.get("method_name") == "do_inference"
            and (decision.get("mock") is True if self.mode == "mock" else "mock" not in decision)
        ):
            raise PolicyError(f"Guardian did not return an explicit matching {self.mode} allow decision")


def local_asset_did(fl_ctx) -> str:
    config = fl_ctx.get_prop(FLContextKey.CLIENT_CONFIG) or {}
    asset_did = config.get("site_config", {}).get("asset_did")
    if not isinstance(asset_did, str) or not asset_did.strip():
        raise PolicyError("site has no configured asset_did")
    return asset_did


def site_route(job_meta, fl_ctx, app_name=None):
    asset_did = local_asset_did(fl_ctx)
    custom = job_meta.get(JobMetaKey.CUSTOM_PROPS, {})
    route = custom.get(ROUTES, {}).get(fl_ctx.get_identity_name())
    if not isinstance(route, dict) or route.get("asset_did") != asset_did:
        raise PolicyError("missing capability route or asset_did does not match local configuration")
    if app_name is not None and route.get("app_name") != app_name:
        raise PolicyError("capability route targets a different application")
    capability = route.get("capability_b64")
    if not isinstance(capability, str) or not capability.strip():
        raise PolicyError("missing capability for this site")
    return route


class PolicyRegistrationRouter(FLComponent):
    """Resolve asset DIDs from registered clients, never infer them from site names."""

    def handle_event(self, event_type, fl_ctx):
        if event_type != EventType.BEFORE_CHECK_CLIENT_RESOURCES:
            return
        meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        try:
            custom = meta.setdefault(JobMetaKey.CUSTOM_PROPS, {})
            # Ignore any caller-supplied routes; rebuild from this registration snapshot.
            custom[ROUTES] = {}
            capabilities = custom.get(CAPABILITIES, {})
            engine = fl_ctx.get_engine()
            for site in fl_ctx.get_prop(FLContextKey.CLIENT_RESOURCE_SPECS, {}):
                client = engine.get_client_from_name(site)
                asset_did = (client.get_site_config() or {}).get("asset_did") if client else None
                apps = [app for app, sites in meta[JobMetaKey.DEPLOY_MAP].items() if site in sites or "@ALL" in sites]
                if not isinstance(asset_did, str) or not asset_did or len(apps) != 1:
                    raise PolicyError(f"{site}: missing asset_did or ambiguous application")
                capability = capabilities.get(asset_did, {}).get(apps[0])
                # A missing capability is left for the site's authorization hook to deny.
                custom[ROUTES][site] = {
                    "asset_did": asset_did,
                    "app_name": apps[0],
                    "capability_b64": capability,
                }
                self.log_info(fl_ctx, f"POLICY_ROUTE site={site} asset_did={asset_did} app={apps[0]}")
        except Exception as exc:
            fl_ctx.set_prop(FLContextKey.JOB_BLOCK_REASON, f"Policy routing failed: {exc}", sticky=False)


class PolicySecurityHandler(FLComponent):
    """Early metadata gate, using the standard job-level authorization event."""

    def handle_event(self, event_type, fl_ctx):
        if event_type != EventType.AUTHORIZE_COMMAND_CHECK:
            return
        if fl_ctx.get_prop(FLContextKey.COMMAND_NAME) not in ("check_resources", "submit_job"):
            return
        try:
            items = fl_ctx.get_prop(FLContextKey.SECURITY_ITEMS) or {}
            site_route(items.get(FLContextKey.JOB_META) or {}, fl_ctx)
        except Exception as exc:
            # Never set True: an allow must not overwrite another handler's deny.
            fl_ctx.set_prop(FLContextKey.AUTHORIZATION_RESULT, False, sticky=False)
            fl_ctx.set_prop(FLContextKey.AUTHORIZATION_REASON, f"Policy Fabric: {exc}", sticky=False)


class PolicyAppDeployer(AppDeployer):
    """Measure and verify the exact received bytes before the normal deployer unpacks them."""

    def __init__(self, guardian_url, timeout=5.0, guardian_mode="pdo"):
        self.guardian = GuardianClient(guardian_url, timeout, mode=guardian_mode)

    def deploy(self, workspace, job_id, job_meta, app_name, app_data, fl_ctx):
        try:
            route = site_route(job_meta, fl_ctx, app_name)
            digest = hash_bytes(app_data)
            self.guardian.verify(route["capability_b64"], digest)
        except Exception as exc:
            return f"Policy Fabric denied deployment: {exc}"
        # Retain NVFlare's own scope, application-validation, and authorization checks.
        return super().deploy(workspace, job_id, job_meta, app_name, app_data, fl_ctx)


class FrozenAppJobManager(SimpleJobDefManager):
    """Preserve a pre-approved archive through NVFlare's unpack/repack storage path."""

    def get_app(self, job, app_name, fl_ctx):
        ordinary_archive = super().get_app(job, app_name, fl_ctx)
        participants = job.get_deployment()[app_name]
        if participants == ["server"]:
            return ordinary_archive
        # The ordinary app remains visible to NVFlare's submit-time config validation.
        # Only this inner immutable archive is actually sent/executed at clients.
        with zipfile.ZipFile(io.BytesIO(ordinary_archive)) as archive:
            if archive.namelist().count(PAYLOAD_FILE) != 1:
                raise PolicyError(f"{app_name}: expected one frozen {PAYLOAD_FILE}")
            return archive.read(PAYLOAD_FILE)
