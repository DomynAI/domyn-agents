import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "prune_alpha_releases.py"
spec = importlib.util.spec_from_file_location("prune_alpha_releases", SCRIPT)
assert spec is not None and spec.loader is not None
prune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prune)

ALPHAS = [f"v1.6.3a{n}" for n in range(1, 9)]


class TestAfterAnAlpha:
    def test_the_newest_five_stay(self) -> None:
        released = ["v1.6.2", *ALPHAS]
        assert prune.to_delete(released, "v1.6.3a8") == ["v1.6.3a1", "v1.6.3a2", "v1.6.3a3"]

    def test_ordered_by_number_not_by_text(self) -> None:
        released = ["v1.6.3a9", "v1.6.3a10", "v1.6.3a11", "v1.6.3a12", "v1.6.3a13", "v1.6.3a2"]
        assert prune.to_delete(released, "v1.6.3a13") == ["v1.6.3a2"]

    def test_fewer_than_five_stay_all(self) -> None:
        assert prune.to_delete(["v1.6.3a1", "v1.6.3a2"], "v1.6.3a2") == []


class TestAfterAStableRelease:
    def test_the_alphas_before_it_go(self) -> None:
        released = ["v1.6.2", "v1.6.3a1", "v1.6.3a2", "v1.6.3"]
        assert prune.to_delete(released, "v1.6.3") == ["v1.6.3a1", "v1.6.3a2"]

    def test_the_alphas_of_a_later_version_stay(self) -> None:
        assert prune.to_delete(["v1.6.4a1", "v1.6.3a1"], "v1.6.3") == ["v1.6.3a1"]

    def test_other_releases_are_never_touched(self) -> None:
        released = ["v1.6.2", "v1.6.3rc1", "v1.6.3-rc3", "v1.6.3"]
        assert prune.to_delete(released, "v1.6.3") == []


class TestAfterAReleaseCandidate:
    def test_nothing_goes(self) -> None:
        assert prune.to_delete(["v1.6.3a1", "v1.6.3a2"], "v1.6.3rc1") == []
