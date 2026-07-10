"""Shared test isolation.

The proxy tags requests with the "current project" and can inject a Project
Primer, both of which read/write files under the real ~/.tokensnap and may
shell out to git against the real project. This autouse fixture redirects that
state to a throwaway location for *every* test, so the suite never scans the
real project, spawns git on it, or overwrites the user's files. Tests that
exercise these features directly still monkeypatch their own temp paths on top.
"""

import pytest

from tokensnap import project, project_primer


@pytest.fixture(autouse=True)
def _isolate_tokensnap_project_state(tmp_path_factory, monkeypatch):
    base = tmp_path_factory.mktemp("tsnap-state")
    monkeypatch.setattr(project, "PROJECT_FILE", base / "current_project")
    monkeypatch.setattr(project_primer, "LAST_CARD_FILE", base / "last_project_card.json")
    monkeypatch.delenv("TOKENSNAP_PROJECT", raising=False)
    project_primer.reset_cache()
    # Project Cortex keeps per-session primed/tracked state on the proxy module;
    # clear it so sessions don't leak between tests.
    from tokensnap import proxy
    proxy.reset_cortex_state()
    yield
    project_primer.reset_cache()
    proxy.reset_cortex_state()
