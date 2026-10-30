import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "publish_docs_version.py"
spec = importlib.util.spec_from_file_location("publish_docs_version", SCRIPT)
assert spec is not None and spec.loader is not None
publish_docs_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish_docs_version)
publish = publish_docs_version.publish


def make_build(root: Path, marker: str) -> Path:
    build = root / f"build-{marker}"
    (build / "guides").mkdir(parents=True)
    (build / "index.html").write_text(marker)
    (build / "guides" / "tools.html").write_text(marker)
    return build


def listed(site: Path) -> dict:
    return json.loads((site / "versions.json").read_text())


class TestPublishingAVersion:
    def test_the_build_lands_under_its_tag(self, tmp_path: Path) -> None:
        site = tmp_path / "site"
        publish(site, make_build(tmp_path, "one"), "v1.0.0", "/repo/")

        assert (site / "v1.0.0" / "guides" / "tools.html").read_text() == "one"
        assert (site / ".nojekyll").exists()
        assert listed(site) == {"latest": "v1.0.0", "versions": ["v1.0.0"]}

    def test_the_root_goes_to_the_latest_version(self, tmp_path: Path) -> None:
        site = tmp_path / "site"
        publish(site, make_build(tmp_path, "one"), "v1.0.0", "/repo/")
        publish(site, make_build(tmp_path, "two"), "v1.1.0", "/repo/")

        assert "/repo/v1.1.0/" in (site / "index.html").read_text()
        assert '"v1.1.0"' in (site / "404.html").read_text()
        assert (site / "v1.0.0" / "index.html").read_text() == "one"

    def test_a_patch_of_an_older_line_does_not_take_the_root(self, tmp_path: Path) -> None:
        site = tmp_path / "site"
        publish(site, make_build(tmp_path, "new"), "v1.10.0")
        publish(site, make_build(tmp_path, "old"), "v1.9.5")

        assert listed(site) == {"latest": "v1.10.0", "versions": ["v1.10.0", "v1.9.5"]}

    def test_a_prerelease_is_published_but_not_the_default(self, tmp_path: Path) -> None:
        site = tmp_path / "site"
        publish(site, make_build(tmp_path, "one"), "v1.0.0")
        publish(site, make_build(tmp_path, "rc"), "v1.1.0-rc1")

        assert listed(site) == {"latest": "v1.0.0", "versions": ["v1.1.0-rc1", "v1.0.0"]}
        assert (site / "v1.1.0-rc1" / "index.html").exists()

    def test_the_release_outranks_its_own_prerelease(self, tmp_path: Path) -> None:
        site = tmp_path / "site"
        publish(site, make_build(tmp_path, "rc"), "v2.0.0-rc1")
        publish(site, make_build(tmp_path, "final"), "v2.0.0")

        assert listed(site)["latest"] == "v2.0.0"

    def test_republishing_a_tag_replaces_it_without_leftovers(self, tmp_path: Path) -> None:
        site = tmp_path / "site"
        publish(site, make_build(tmp_path, "first"), "v1.0.0")
        (site / "v1.0.0" / "stale.html").write_text("stale")
        publish(site, make_build(tmp_path, "second"), "v1.0.0")

        assert (site / "v1.0.0" / "index.html").read_text() == "second"
        assert not (site / "v1.0.0" / "stale.html").exists()
        assert listed(site)["versions"] == ["v1.0.0"]


class TestRefusingWhatIsNotAVersion:
    @pytest.mark.parametrize("version", ["main", "1.0.0", "v1.0", "v1.0.0/../x", "../v1.0.0"])
    def test_a_name_that_is_not_a_release_tag(self, tmp_path: Path, version: str) -> None:
        with pytest.raises(ValueError, match="release tag"):
            publish(tmp_path / "site", make_build(tmp_path, "x"), version)

    def test_a_directory_that_is_not_a_build(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="Docusaurus build"):
            publish(tmp_path / "site", tmp_path / "empty", "v1.0.0")

    @pytest.mark.parametrize("base", ["repo", "/repo", "repo/"])
    def test_a_base_without_both_slashes(self, tmp_path: Path, base: str) -> None:
        with pytest.raises(ValueError, match="--base"):
            publish(tmp_path / "site", make_build(tmp_path, "x"), "v1.0.0", base)
