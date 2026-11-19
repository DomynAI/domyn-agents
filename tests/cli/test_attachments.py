"""Attaching images in the CLI, without a terminal: which pastes are images, where the
files go, the clipboard, and what the model is told."""

import os
from pathlib import Path

import pytest
from PIL import Image

from domyn.agents.interfaces.cli import attachments as attachments_mod
from domyn.agents.interfaces.cli.attachments import (
    MAX_PASTED_PATHS,
    NOT_READABLE,
    AttachmentError,
    Attachments,
    Place,
    image_paths,
)


def _picture(path: Path, size: tuple[int, int] = (8, 8)) -> Path:
    Image.new("RGB", size, "red").save(path)
    return path


@pytest.fixture
def place(tmp_path: Path) -> Place:
    state = tmp_path / "state"
    state.mkdir()
    return Place(state=state, folder=state / "conversations" / "c1" / "attachments")


@pytest.fixture
def attachments(place: Place) -> Attachments:
    return Attachments(lambda: place)


class TestWhichPastesAreImages:
    def test_a_plain_path(self, tmp_path: Path) -> None:
        shot = _picture(tmp_path / "shot.png")
        assert image_paths(str(shot)) == [shot]

    def test_escaped_spaces_as_a_terminal_drags_them_in(self, tmp_path: Path) -> None:
        shot = _picture(tmp_path / "My Shot.png")
        pasted = str(shot).replace(" ", "\\ ")
        assert image_paths(pasted) == [shot]

    def test_quoted(self, tmp_path: Path) -> None:
        shot = _picture(tmp_path / "My Shot.png")
        assert image_paths(f"'{shot}' ") == [shot]

    def test_a_file_url(self, tmp_path: Path) -> None:
        shot = _picture(tmp_path / "My Shot.png")
        assert image_paths(shot.as_uri()) == [shot]

    def test_several(self, tmp_path: Path) -> None:
        one = _picture(tmp_path / "one.png")
        two = _picture(tmp_path / "two.jpg")
        assert image_paths(f"{one} {two}\n") == [one, two]

    def test_a_file_that_is_not_an_image_is_text(self, tmp_path: Path) -> None:
        source = tmp_path / "app.py"
        source.write_text("print('hi')\n", encoding="utf-8")
        assert image_paths(str(source)) is None

    def test_one_entry_that_is_not_an_image_makes_the_whole_paste_text(
        self, tmp_path: Path
    ) -> None:
        shot = _picture(tmp_path / "shot.png")
        assert image_paths(f"{shot} {tmp_path / 'missing.png'}") is None

    @pytest.mark.parametrize(
        "pasted",
        ["", "   ", "look at this", "it's unbalanced 'quote", "/no/such/file.png"],
    )
    def test_text(self, pasted: str) -> None:
        assert image_paths(pasted) is None

    def test_a_folder(self, tmp_path: Path) -> None:
        assert image_paths(str(tmp_path)) is None

    def test_too_many_entries_is_text(self, tmp_path: Path) -> None:
        shot = _picture(tmp_path / "shot.png")
        assert image_paths(" ".join([str(shot)] * (MAX_PASTED_PATHS + 1))) is None


class TestAttachingAFile:
    def test_it_is_copied_into_the_conversation(
        self, attachments: Attachments, place: Place, tmp_path: Path
    ) -> None:
        shot = _picture(tmp_path / "shot.PNG")
        assert attachments.add_file(shot) == "[Image #1]"
        [copy] = place.folder.iterdir()
        assert copy.name == "shot.png"
        assert copy.read_bytes() == shot.read_bytes()

    def test_numbers_go_up_and_names_do_not_clash(
        self, attachments: Attachments, place: Place, tmp_path: Path
    ) -> None:
        shot = _picture(tmp_path / "shot.png")
        assert attachments.add_file(shot) == "[Image #1]"
        assert attachments.add_file(shot) == "[Image #2]"
        assert attachments.add_file(shot) == "[Image #3]"
        assert sorted(p.name for p in place.folder.iterdir()) == [
            "shot-2.png",
            "shot-3.png",
            "shot.png",
        ]

    @pytest.mark.parametrize(
        ("name", "copied"),
        [
            ("My Shot [final].png", "My_Shot_final.png"),
            ("error-dialog_v2.png", "error-dialog_v2.png"),
            ("schermata è.png", "schermata_è.png"),
            ("schermata e\u0300.png", "schermata_\u00e8.png"),
            (".hidden.png", "hidden.png"),
            ("[].png", "image.png"),
            ("shot.p g", "shot.png"),
        ],
    )
    def test_the_name_is_kept_as_far_as_it_is_safe(
        self, attachments: Attachments, place: Place, tmp_path: Path, name: str, copied: str
    ) -> None:
        """The path is written into the message, where a `]` or a new line would
        break it, so only letters, digits, `-`, `_` and `.` stay."""
        Image.new("RGB", (4, 4)).save(tmp_path / name, "PNG")
        attachments.add_file(tmp_path / name)
        [copy] = place.folder.iterdir()
        assert copy.name == copied

    def test_a_long_name_is_cut(
        self, attachments: Attachments, place: Place, tmp_path: Path
    ) -> None:
        shot = _picture(tmp_path / f"{'a' * 150}.png")
        attachments.add_file(shot)
        [copy] = place.folder.iterdir()
        assert copy.name == f"{'a' * attachments_mod.MAX_NAME_CHARS}.png"

    def test_a_file_without_an_extension_is_named_by_its_format(
        self, attachments: Attachments, place: Place, tmp_path: Path
    ) -> None:
        shot = tmp_path / "shot"
        Image.new("RGB", (4, 4)).save(shot, "JPEG")
        attachments.add_file(shot)
        [copy] = place.folder.iterdir()
        assert copy.name == "shot.jpeg"

    def test_not_an_image(self, attachments: Attachments, tmp_path: Path) -> None:
        notes = tmp_path / "notes.md"
        notes.write_text("# Notes\n", encoding="utf-8")
        with pytest.raises(AttachmentError, match="not an image"):
            attachments.add_file(notes)

    def test_not_a_file(self, attachments: Attachments, tmp_path: Path) -> None:
        with pytest.raises(AttachmentError, match="not a file"):
            attachments.add_file(tmp_path / "missing.png")

    def test_too_big(
        self, attachments: Attachments, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(attachments_mod, "MAX_ATTACHED_BYTES", 10)
        with pytest.raises(AttachmentError, match="over"):
            attachments.add_file(_picture(tmp_path / "shot.png"))

    def test_refused_when_the_agent_cannot_look_at_images(self, tmp_path: Path) -> None:
        nowhere = Attachments(lambda: None)
        with pytest.raises(AttachmentError) as refused:
            nowhere.add_file(_picture(tmp_path / "shot.png"))
        assert str(refused.value) == NOT_READABLE

    @pytest.mark.skipif(os.name == "nt", reason="symlinks")
    def test_never_through_a_link(
        self, attachments: Attachments, place: Place, tmp_path: Path
    ) -> None:
        """A project folder used by its path can be someone else's: a link below its
        state folder would put the copy wherever it points."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (place.state / "conversations").symlink_to(elsewhere)
        with pytest.raises(AttachmentError, match="is a link"):
            attachments.add_file(_picture(tmp_path / "shot.png"))
        assert list(elsewhere.iterdir()) == []

    def test_the_place_is_asked_every_time(self, tmp_path: Path) -> None:
        """``/clear`` and ``/resume`` change the conversation, so the folder."""
        folders = iter([tmp_path / "a", tmp_path / "b"])
        attachments = Attachments(lambda: Place(state=tmp_path, folder=next(folders)))
        shot = _picture(tmp_path / "shot.png")
        attachments.add_file(shot)
        attachments.add_file(shot)
        assert len(list((tmp_path / "a").iterdir())) == 1
        assert len(list((tmp_path / "b").iterdir())) == 1


class TestTheClipboard:
    @pytest.fixture(autouse=True)
    def no_finder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The real one asks osascript what the clipboard holds.
        monkeypatch.setattr(attachments_mod, "_clipboard_file", lambda: None)

    def test_an_image_is_saved_as_png(
        self, attachments: Attachments, place: Place, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            attachments_mod.ImageGrab, "grabclipboard", lambda: Image.new("CMYK", (4, 4))
        )
        assert attachments.from_clipboard() == ["[Image #1]"]
        assert attachments.from_clipboard() == ["[Image #2]"]
        assert sorted(p.name for p in place.folder.iterdir()) == ["pasted-1.png", "pasted-2.png"]
        with Image.open(place.folder / "pasted-1.png") as image:
            assert image.format == "PNG"

    def test_files_copied_in_a_file_manager(
        self,
        attachments: Attachments,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        shot = _picture(tmp_path / "shot.png")
        notes = tmp_path / "notes.md"
        notes.write_text("# Notes\n", encoding="utf-8")
        monkeypatch.setattr(
            attachments_mod.ImageGrab, "grabclipboard", lambda: [str(shot), str(notes)]
        )
        assert attachments.from_clipboard() == ["[Image #1]"]

    def test_a_file_copied_in_finder_wins_over_its_icon(
        self,
        attachments: Attachments,
        place: Place,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        shot = _picture(tmp_path / "shot.png", size=(16, 16))
        monkeypatch.setattr(attachments_mod.sys, "platform", "darwin")
        monkeypatch.setattr(attachments_mod, "_clipboard_file", lambda: shot)
        monkeypatch.setattr(
            attachments_mod.ImageGrab, "grabclipboard", lambda: Image.new("RGB", (1, 1))
        )
        attachments.from_clipboard()
        [copy] = place.folder.iterdir()
        assert copy.read_bytes() == shot.read_bytes()

    def test_nothing_in_it(self, attachments: Attachments, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(attachments_mod.ImageGrab, "grabclipboard", lambda: None)
        with pytest.raises(AttachmentError, match="No image in the clipboard"):
            attachments.from_clipboard()

    def test_linux_without_the_commands(
        self, attachments: Attachments, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing() -> None:
            raise NotImplementedError("wl-paste or xclip is required")

        monkeypatch.setattr(attachments_mod.ImageGrab, "grabclipboard", missing)
        with pytest.raises(AttachmentError, match="wl-paste or xclip"):
            attachments.from_clipboard()

    def test_a_failure_says_so_without_the_detail(
        self, attachments: Attachments, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken() -> None:
            raise ChildProcessError("xclip error: something private")

        monkeypatch.setattr(attachments_mod.ImageGrab, "grabclipboard", broken)
        with pytest.raises(AttachmentError) as failed:
            attachments.from_clipboard()
        assert "private" not in str(failed.value)

    def test_the_clipboard_is_not_read_when_nothing_could_look_at_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unexpected() -> None:
            raise AssertionError("the clipboard was read")

        monkeypatch.setattr(attachments_mod.ImageGrab, "grabclipboard", unexpected)
        with pytest.raises(AttachmentError, match="cannot look at images"):
            Attachments(lambda: None).from_clipboard()


class TestWhatTheModelIsTold:
    def test_a_chip_names_the_file(self, attachments: Attachments, tmp_path: Path) -> None:
        chip = attachments.add_file(_picture(tmp_path / "shot.png"))
        sent = attachments.expand(f"{chip} what does it say?")
        assert sent.startswith("[Image #1, attached by the user: ")
        assert "read_image" in sent
        assert sent.endswith(" what does it say?")
        path = sent.split("attached by the user: ", 1)[1].split(". Look", 1)[0]
        assert Path(path).is_file()

    def test_a_chip_typed_by_hand_is_left_alone(self, attachments: Attachments) -> None:
        assert attachments.expand("see [Image #3]") == "see [Image #3]"

    def test_text_without_chips_is_unchanged(self, attachments: Attachments) -> None:
        assert attachments.expand("no images here") == "no images here"

    def test_a_new_conversation_starts_again_at_one(
        self, attachments: Attachments, tmp_path: Path
    ) -> None:
        shot = _picture(tmp_path / "shot.png")
        attachments.add_file(shot)
        attachments.reset()
        assert attachments.expand("[Image #1]") == "[Image #1]"
        assert attachments.add_file(shot) == "[Image #1]"
