import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LICENCES = ROOT / "licences"


def _sources() -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for line in (LICENCES / "SOURCES").read_text(encoding="utf-8").splitlines():
        digest, path, url = line.split("  ")
        entries[path] = (digest, url)
    return entries


def test_every_licence_text_matches_its_pinned_source():
    sources = _sources()
    shipped = {
        path.relative_to(LICENCES).as_posix()
        for path in LICENCES.rglob("*")
        if path.is_file() and path.name != "SOURCES"
    }
    assert shipped == set(sources)
    for path, (digest, url) in sources.items():
        assert hashlib.sha256((LICENCES / path).read_bytes()).hexdigest() == digest, path
        commit = url.split("/")[5]
        assert len(commit) == 40, f"{path} is not fetched at a commit: {url}"
