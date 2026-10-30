import importlib.util
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "tag_release.py"
spec = importlib.util.spec_from_file_location("tag_release", SCRIPT)
assert spec is not None and spec.loader is not None
tag_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tag_release)


class TestTheLatestRelease:
    def test_is_the_highest_stable_tag(self) -> None:
        tags = ["v1.6.2", "v1.10.0", "v1.9.9", "v1.6.3-rc3", "v2.0.0rc1"]
        assert tag_release.latest_stable(tags) == (1, 10, 0)

    def test_ignores_anything_else(self) -> None:
        assert tag_release.latest_stable(["v1.5.4rc1-browser", "rust-v1.0.0"]) is None


class TestTheNextVersion:
    @pytest.mark.parametrize(
        ("spec", "expected"), [("patch", "1.6.3"), ("minor", "1.7.0"), ("major", "2.0.0")]
    )
    def test_a_bump_counts_from_the_latest(self, spec: str, expected: str) -> None:
        assert tag_release.next_version(spec, (1, 6, 2)) == expected

    def test_the_first_release(self) -> None:
        assert tag_release.next_version("minor", None) == "0.1.0"

    @pytest.mark.parametrize("version", ["1.7.0", "1.7.0rc1", "1.7.0a2", "1.6.3b1", "2.0.0"])
    def test_a_version_given_in_full(self, version: str) -> None:
        assert tag_release.next_version(version, (1, 6, 2)) == version

    @pytest.mark.parametrize("version", ["1.7.0-rc1", "v1.7.0", "1.7", "1.7.0.dev1", "next"])
    def test_only_the_pep_440_spelling(self, version: str) -> None:
        with pytest.raises(ValueError, match="not patch, minor, major"):
            tag_release.next_version(version, (1, 6, 2))

    @pytest.mark.parametrize("version", ["1.6.2", "1.6.1", "1.6.2rc1"])
    def test_never_at_or_before_the_latest(self, version: str) -> None:
        with pytest.raises(ValueError, match=re.escape("not after the latest release, v1.6.2")):
            tag_release.next_version(version, (1, 6, 2))


class TestTheNextAlpha:
    def test_the_first_after_a_release(self) -> None:
        assert tag_release.next_alpha(["v1.6.2", "v1.6.1"], (1, 6, 2)) == "1.6.3a1"

    def test_numbered_after_the_ones_before(self) -> None:
        tags = ["v1.6.2", "v1.6.3a1", "v1.6.3a2", "v1.6.3a10"]
        assert tag_release.next_alpha(tags, (1, 6, 2)) == "1.6.3a11"

    def test_alphas_of_an_older_patch_do_not_count(self) -> None:
        tags = ["v1.6.2a7", "v1.6.2", "v1.6.3rc1"]
        assert tag_release.next_alpha(tags, (1, 6, 2)) == "1.6.3a1"

    def test_with_no_release_yet(self) -> None:
        assert tag_release.next_alpha([], None) == "0.0.1a1"
