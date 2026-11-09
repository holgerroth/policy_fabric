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

"""Example-local prototype of a generic ProdEnv callback before submission.

No installed NVFlare code is patched. Policy-specific behavior belongs in the
callback, not in this module. See NVFLARE_HOOK_PROPOSAL.md for the core proposal.
"""

import inspect
import tempfile
from pathlib import Path
from typing import Callable, Optional

from nvflare.fuel.utils.job_secret_scanner import warn_on_potential_secrets_in_job_dir
from nvflare.job_config.api import FedJob
from nvflare.recipe import ProdEnv
from nvflare.recipe.session_mgr import SessionManager

BeforeSubmit = Callable[[Path], None]


class _HookedSessionManager(SessionManager):
    def __init__(self, session_params, before_submit: BeforeSubmit):
        super().__init__(session_params)
        self.before_submit = before_submit

    def submit_job(self, job: FedJob) -> str:
        # Mirrors SessionManager's export/submit lifecycle, adding one callback.
        # The hook sees the final Recipe-generated files, including env settings.
        with tempfile.TemporaryDirectory() as temp_dir:
            job.export_job(temp_dir)
            job_path = Path(temp_dir) / job.name
            result = self.before_submit(job_path)
            if result is not None:
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("before_submit must return None on success or raise to prevent submission")
            warn_on_potential_secrets_in_job_dir(temp_dir, job_name=job.name)
            # Do not connect to the federation until preparation succeeds.
            sess = self._get_session()
            try:
                job_id = sess.submit_job(str(job_path))
            finally:
                sess.close()
        print(f"Submitted job '{job.name}' with ID: {job_id}")
        return job_id


class PolicyProdEnv(ProdEnv):
    """Production environment for this policy example, with a generic submission hook.

    The callback runs once per submission attempt, edits files in place, and must
    return None. An exception prevents submission. The folder is temporary and
    is removed after submission or failure. Without a callback, use the original
    SessionManager unchanged. Monitoring and results remain inherited from ProdEnv.

    This compatibility subclass uses private session-manager wiring; it is a
    prototype for a public core hook, not an existing NVFlare core API.
    """

    def __init__(self, *args, before_submit: Optional[BeforeSubmit] = None, **kwargs):
        if before_submit is not None:
            if not callable(before_submit) or inspect.iscoroutinefunction(before_submit):
                raise TypeError("before_submit must be a synchronous callable or None")
        super().__init__(*args, **kwargs)
        self.before_submit = before_submit

    def _get_session_manager(self):
        manager = super()._get_session_manager()
        if self.before_submit is not None and not isinstance(manager, _HookedSessionManager):
            self._session_manager = _HookedSessionManager(manager.session_params, self.before_submit)
        return self._session_manager
