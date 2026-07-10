"""Offline tests for the Project Cortex CLI commands: `tokensnap focus` and
`tokensnap dna`. They act on the tagged project directory (isolated to a temp
dir here) and never touch the network.
"""

import pytest
from typer.testing import CliRunner

from tokensnap import cli
from tokensnap import project as project_mod
from tokensnap import project_dna

runner = CliRunner()


@pytest.fixture(autouse=True)
def temp_project(tmp_path, monkeypatch):
    monkeypatch.setattr(project_mod, "PROJECT_FILE", tmp_path / "current_project")
    monkeypatch.delenv("TOKENSNAP_PROJECT", raising=False)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "pyproject.toml").write_text('dependencies = ["flask"]\n', encoding="utf-8")
    (proj / "main.py").write_text("print(1)\n", encoding="utf-8")
    project_mod.set_current_project(str(proj))
    return str(proj)


def test_focus_set_and_show(temp_project):
    res = runner.invoke(cli.app, ["focus", "Ship", "Project", "Cortex"])
    assert res.exit_code == 0
    assert project_dna.load_dna(temp_project)["focus"] == "Ship Project Cortex"
    shown = runner.invoke(cli.app, ["focus"])
    assert shown.exit_code == 0
    assert "Ship Project Cortex" in shown.output


def test_dna_shows_stack(temp_project):
    res = runner.invoke(cli.app, ["dna"])
    assert res.exit_code == 0
    # The DNA file is generated on demand.
    assert project_dna.dna_path(temp_project).exists()
    assert project_dna.load_dna(temp_project)["static"]["language"] == "Python"


def test_dna_refresh_rescans(temp_project):
    res = runner.invoke(cli.app, ["dna", "--refresh"])
    assert res.exit_code == 0
    dna = project_dna.load_dna(temp_project)
    assert dna["static"]["framework"] == "Flask"
    assert "main.py" in dna["static"]["entry_points"]
