"""``local_dir``: ordered roots, the first primary, read through the file-store view."""

from pathlib import Path

import pytest

from domyn.agents.resources import (
    LocalDirResource,
    ResourceError,
    relative_path,
    resolve_resource,
)


def _resource(*roots: Path, read_only: bool = False) -> LocalDirResource:
    return LocalDirResource(id="files", paths=[str(r) for r in roots], read_only=read_only)


class TestRelativePaths:
    @pytest.mark.parametrize("path", ["/etc/passwd", "~/x", "..", "../x", "a/../../x"])
    def test_a_path_leaving_the_resource_is_refused(self, path: str) -> None:
        with pytest.raises(ResourceError):
            relative_path(path)

    @pytest.mark.parametrize(
        ("path", "normal"), [("", ""), (".", ""), ("a/./b", "a/b"), ("a/b/../c", "a/c")]
    )
    def test_a_path_inside_it_is_normalized(self, path: str, normal: str) -> None:
        assert relative_path(path) == normal


class TestDeclaring:
    def test_it_is_resolved_by_its_type(self) -> None:
        resource = resolve_resource({"id": "a", "type": "local_dir", "paths": "."})
        assert isinstance(resource, LocalDirResource)
        assert resource.paths == ["."]

    def test_an_unknown_type_is_refused_naming_the_known_ones(self) -> None:
        with pytest.raises(ResourceError, match=r"unknown type 'nfs'.*local_dir"):
            resolve_resource({"id": "a", "type": "nfs"})

    def test_an_unknown_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="workspace"):
            resolve_resource({"id": "a", "type": "local_dir", "paths": ["."], "workspace": "."})

    @pytest.mark.parametrize("paths", [[], [""]])
    def test_it_needs_a_root(self, paths: list[str]) -> None:
        with pytest.raises(ValueError):
            LocalDirResource(id="a", paths=paths)

    def test_a_missing_root_is_a_warning_not_an_error(self, tmp_path: Path) -> None:
        warnings = _resource(tmp_path, tmp_path / "missing").check()
        assert warnings == [f"resource 'files': {tmp_path / 'missing'} does not exist"]

    def test_a_relative_root_stays_relative(self) -> None:
        # The skills and instructions readers check every folder of a relative path
        # for links, which an absolute one would skip.
        assert LocalDirResource(id="a", paths=[".agents"]).local_roots() == (Path(".agents"),)


class TestTheFileStoreView:
    @pytest.mark.asyncio
    async def test_reading_takes_the_first_root_holding_the_file(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        (first / "x").mkdir(parents=True)
        (second / "x").mkdir(parents=True)
        (first / "x" / "same.md").write_text("first")
        (second / "x" / "same.md").write_text("second")
        (second / "x" / "only.md").write_text("only")
        resource = _resource(first, second)

        assert await resource.read_text("x/same.md") == "first"
        assert await resource.read_text("x/only.md") == "only"
        assert [e.path for e in await resource.list("x")] == ["x/only.md", "x/same.md"]

    @pytest.mark.asyncio
    async def test_a_missing_file_is_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await _resource(tmp_path).read("nothing.md")

    @pytest.mark.asyncio
    async def test_writing_goes_to_the_primary_root(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        await _resource(first, second).write("deep/new.md", b"hi")

        assert (first / "deep" / "new.md").read_text() == "hi"
        assert not (second / "deep").exists()

    @pytest.mark.asyncio
    async def test_a_read_only_resource_refuses_to_write(self, tmp_path: Path) -> None:
        with pytest.raises(ResourceError, match="read-only"):
            await _resource(tmp_path, read_only=True).write("x.md", b"hi")
        assert not (tmp_path / "x.md").exists()

    @pytest.mark.asyncio
    async def test_a_link_is_not_followed(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret").write_text("no")
        root = tmp_path / "root"
        root.mkdir()
        (root / "away").symlink_to(outside)

        resource = _resource(root)
        with pytest.raises(ResourceError, match="link"):
            await resource.read("away/secret")
        with pytest.raises(ResourceError, match="link"):
            await resource.write("away/planted", b"x")
        assert [e.path for e in await resource.list()] == []
        assert not (outside / "planted").exists()

    @pytest.mark.asyncio
    async def test_a_climbing_path_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ResourceError, match="outside"):
            await _resource(tmp_path / "root").read("../x")

    @pytest.mark.asyncio
    async def test_a_root_that_is_a_file_holds_that_file(self, tmp_path: Path) -> None:
        soul = tmp_path / "SOUL.md"
        soul.write_text("me")
        resource = _resource(soul)
        assert await resource.read_text("SOUL.md") == "me"
        assert [e.path for e in await resource.list()] == ["SOUL.md"]

    @pytest.mark.asyncio
    async def test_stat(self, tmp_path: Path) -> None:
        (tmp_path / "d").mkdir()
        (tmp_path / "d" / "f").write_text("abc")
        resource = _resource(tmp_path)
        entry = await resource.stat("d/f")
        assert entry is not None and not entry.is_dir and entry.size == 3
        folder = await resource.stat("d")
        assert folder is not None and folder.is_dir
        assert await resource.stat("nope") is None


class TestTheMountView:
    @pytest.mark.asyncio
    async def test_a_local_folder_is_its_own_mount(self, tmp_path: Path) -> None:
        spec = await _resource(tmp_path, read_only=True).materialize()
        assert spec.paths == (tmp_path.resolve(),)
        assert spec.read_only
