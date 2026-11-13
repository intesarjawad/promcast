from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_publication_requires_the_complete_check_workflow_at_the_tagged_commit():
    release = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    checks = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    gate = release["jobs"]["checks"]
    assert gate["uses"] == "./.github/workflows/ci.yml"
    assert "if" not in gate
    assert "workflow_call" in checks.get("on", checks.get(True))
    for name in ("image", "pypi", "chart", "compose"):
        assert "checks" in release["jobs"][name]["needs"]
        assert "if" not in release["jobs"][name]
    for workflow in (release, checks):
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/checkout@"):
                    assert step["with"]["ref"] == "${{ github.sha }}"
    assert {
        "check",
        "docs",
        "compose-smoke",
        "coldstart",
        "tsdb-compat",
        "helm-chart",
        "bench",
    } <= checks["jobs"].keys()
    for command in ("make check", "make check-slow"):
        assert any(step.get("run") == command for step in checks["jobs"]["check"]["steps"])
    assert any(step.get("run") == "make check-site" for step in checks["jobs"]["docs"]["steps"])
