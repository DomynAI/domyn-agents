import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "coverage_total.py"
spec = importlib.util.spec_from_file_location("coverage_total", SCRIPT)
assert spec is not None and spec.loader is not None
coverage_total = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage_total)


def report(tmp_path: Path, rate: str) -> Path:
    path = tmp_path / "coverage.xml"
    path.write_text(f'<?xml version="1.0" ?><coverage line-rate="{rate}" version="7"></coverage>')
    return path


class TestTheTotal:
    def test_is_the_line_rate_in_percent(self, tmp_path: Path) -> None:
        assert coverage_total.total(report(tmp_path, "0.81154")) == 81.15

    def test_a_report_without_one_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "other.xml"
        path.write_text("<coverage/>")
        with pytest.raises(ValueError, match="no line-rate"):
            coverage_total.total(path)

    def test_printed_with_two_decimals(self, tmp_path: Path, capsys) -> None:
        assert coverage_total.main(["total", str(report(tmp_path, "0.8"))]) == 0
        assert capsys.readouterr().out == "80.00\n"


class TestTheComparison:
    @pytest.mark.parametrize(
        ("current", "ok"),
        [(81.30, True), (81.15, True), (81.05, True), (81.04, False), (79.0, False)],
    )
    def test_it_may_drop_by_the_tolerance_only(self, current: float, ok: bool) -> None:
        assert coverage_total.verdict(current, 81.15, 0.1)[0] is ok

    def test_the_line_says_by_how_much(self) -> None:
        assert coverage_total.verdict(80.65, 81.15, 0.1)[1] == (
            "❌ Total coverage 80.65% (-0.50 against main's 81.15%): it may not drop by more than 0.10."
        )
        assert coverage_total.verdict(81.40, 81.15, 0.1)[1] == (
            "✅ Total coverage 81.40% (+0.25 against main's 81.15%)."
        )

    def test_the_exit_status(self, tmp_path: Path) -> None:
        path = str(report(tmp_path, "0.80"))
        assert coverage_total.main(["compare", path, "--baseline", "80.05"]) == 0
        assert coverage_total.main(["compare", path, "--baseline", "81.00"]) == 1
