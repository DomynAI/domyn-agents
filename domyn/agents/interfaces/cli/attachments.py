"""Images the user attaches to a message: Ctrl-V, a pasted or dragged path.

A terminal hands an application text only: Cmd+V and a bracketed paste carry no
pixels. So the image is fetched here — from the clipboard, or from the file a
pasted path names — and copied into the conversation's ``attachments`` folder
(`SessionPaths.attachments`). The editor shows an ``[Image #N]`` chip where it was
attached, and the message the model gets names the file instead (`Attachments.expand`),
for ``read_image`` to look at when it needs to. No image enters the conversation.

The folder is the conversation's, beside its venv and output: ``domyn clean`` removes
it with them, ``domyn export`` never copies it, and ``--resume`` finds the files again.
``read_image`` may read it for that conversation alone, as its state folder is no root
of the workspace (``images.image_tools``).

Nothing here knows about the editor, so it can be tested without a terminal.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageGrab

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.session import linked_below

CHIP = re.compile(r"\[Image #(\d+)\]")
"""What the editor shows for an attached image."""

MAX_ATTACHED_BYTES = 50_000_000
"""Largest file copied. ``read_image`` scales what it sends; this only stops a
paste from filling the disk with a mistake."""

MAX_PASTE_CHARS = 4096
"""A paste longer than this is text: nobody drags in a list of paths this long."""

MAX_PASTED_PATHS = 20

MAX_NAME_CHARS = 100
"""Longest name kept for an attached copy, before its extension."""

# What an attached copy's name may hold; anything else becomes `_`.
_UNSAFE = re.compile(r"[^\w.-]+")
_SUFFIX = re.compile(r"\.[a-z0-9]+")

NOT_READABLE = (
    "This agent cannot look at images: it needs the files tools and a model that reads "
    "images, which domyn setup, or /model, checks."
)


class AttachmentError(Exception):
    """An image could not be attached. The message is written for the user."""


@dataclass(frozen=True)
class Place:
    """Where the current conversation's attachments go."""

    state: Path
    """The state folder, below which no link is followed (`linked_below`)."""
    folder: Path
    """``<state>/conversations/<id>/attachments``."""


def _image_format(path: Path) -> str | None:
    """Pillow's name for the format of the image at ``path``, or ``None`` when it is
    not one. Reads the header only."""
    try:
        if not path.is_file():
            return None
        with Image.open(path) as image:
            return image.format
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def image_paths(pasted: str) -> list[Path] | None:
    """The image files a paste names, or ``None`` when it is anything else.

    What a terminal pastes for a dragged file, or a file copied in a file manager, is
    its path: with a backslash before each space, quoted, or as a ``file://`` URL. Every entry
    must be an image Pillow opens, or the whole paste is text: a path to a source file
    is pasted as written."""
    text = pasted.strip()
    if not text or len(text) > MAX_PASTE_CHARS:
        return None
    try:
        if os.name == "nt":
            entries = [entry.strip('"') for entry in shlex.split(text, posix=False)]
        else:
            entries = shlex.split(text)
    except ValueError:
        return None
    if not entries or len(entries) > MAX_PASTED_PATHS:
        return None

    paths = []
    for entry in entries:
        if entry.startswith("file://"):
            entry = unquote(urlsplit(entry).path)
        path = Path(entry).expanduser()
        if _image_format(path) is None:
            return None
        paths.append(path.absolute())
    return paths


def _clipboard_file() -> Path | None:
    """The file copied in Finder, if that is what the clipboard holds (macOS).

    Asked before the image: a file copied in Finder also puts its icon on the
    clipboard, which Pillow would hand back as the image."""
    try:
        done = subprocess.run(
            ["osascript", "-e", "POSIX path of (the clipboard as «class furl»)"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    path = done.stdout.strip()
    if done.returncode != 0 or not path:
        return None
    return Path(path)


class Attachments:
    """The images attached in this conversation, by the number of their chip.

    ``place`` says where the current conversation's images go, or ``None`` when the
    agent has no ``read_image`` to look at them with. Asked at every attach, as
    ``/clear``, ``/resume``, a config reload and ``/model`` change the answer.

    The numbers are not forgotten when a message is sent: a draft taken back by the
    editor (the steering reader asks again with it) still holds its chips. They are
    forgotten with the conversation (`reset`).
    """

    def __init__(self, place: Callable[[], Place | None]) -> None:
        self._place = place
        self._paths: dict[int, Path] = {}
        # Each Ctrl-V reads the clipboard in a thread of its own, so two quick
        # presses attach at once: held from picking a free name to numbering the
        # chip, or both would write one file and could take one number.
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Forget every attachment: a new conversation starts at ``[Image #1]``."""
        self._paths.clear()

    def check(self) -> Place:
        """Where an image goes now; raises when none can be attached."""
        place = self._place()
        if place is None:
            raise AttachmentError(NOT_READABLE)
        return place

    def _target(self, place: Place, stem: str, suffix: str, *, numbered: bool = False) -> Path:
        """A file name in the conversation's attachments folder that no other file has.

        ``stem`` is kept, as the name of an image often says what it is
        (``error-dialog.png``), with anything but letters, digits, ``-``, ``_`` and
        ``.`` replaced: the path is written into the message, where a ``]`` or a new
        line would break it. A name already taken gets ``-2``, ``-3``…; ``numbered``
        numbers it from ``-1``, for images that have no name of their own."""
        linked = linked_below(place.state, place.folder)
        if linked is not None:
            raise AttachmentError(f"'{linked}' is a link, so nothing is attached through it.")
        try:
            place.folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            get_logger().debug(f"attachments: could not create the folder, detail: {e}")
            raise AttachmentError(f"'{place.folder}' could not be created.") from None

        # Composed first: macOS may hand a name over with `è` as `e` plus an accent,
        # and the accent alone is no letter.
        stem = unicodedata.normalize("NFC", stem)
        stem = _UNSAFE.sub("_", stem).strip("._-")[:MAX_NAME_CHARS] or "image"
        number = 1 if numbered else None
        while True:
            name = f"{stem}{suffix}" if number is None else f"{stem}-{number}{suffix}"
            target = place.folder / name
            if not target.exists() and not target.is_symlink():
                return target
            number = 2 if number is None else number + 1

    def _register(self, path: Path) -> str:
        number = len(self._paths) + 1
        self._paths[number] = path
        return f"[Image #{number}]"

    def add_file(self, path: Path) -> str:
        """Copy the image at ``path`` into the conversation and return its chip.

        Copied rather than named where it is: a screenshot on the desktop is outside
        the workspace, and the original may have moved by the time a conversation is
        resumed."""
        place = self.check()
        path = path.expanduser()
        if not path.is_file():
            raise AttachmentError(f"'{path}' is not a file.")
        if path.stat().st_size > MAX_ATTACHED_BYTES:
            raise AttachmentError(f"'{path}' is over {MAX_ATTACHED_BYTES // 1_000_000} MB.")
        image_format = _image_format(path)
        if image_format is None:
            raise AttachmentError(f"'{path}' is not an image.")
        suffix = path.suffix.lower()
        if not _SUFFIX.fullmatch(suffix):
            suffix = f".{image_format.lower()}"
        with self._lock:
            target = self._target(place, path.stem, suffix)
            try:
                shutil.copyfile(path, target)
            except OSError as e:
                get_logger().debug(f"attachments: could not copy an image, detail: {e}")
                raise AttachmentError(f"'{path}' could not be copied.") from None
            return self._register(target)

    def add_image(self, image: Image.Image) -> str:
        """Save an image from the clipboard as PNG and return its chip."""
        place = self.check()
        if image.mode not in ("1", "L", "LA", "I", "P", "RGB", "RGBA"):
            image = image.convert("RGBA")
        with self._lock:
            target = self._target(place, "pasted", ".png", numbered=True)
            try:
                image.save(target, "PNG")
            except (OSError, ValueError) as e:
                get_logger().debug(f"attachments: could not save a pasted image, detail: {e}")
                raise AttachmentError("The image in the clipboard could not be saved.") from None
            return self._register(target)

    def from_clipboard(self) -> list[str]:
        """Attach what the clipboard holds: an image, or image files copied in a file
        manager. Blocking (a command on macOS and Linux), so it runs in a thread."""
        self.check()
        if sys.platform == "darwin":
            copied = _clipboard_file()
            if copied is not None:
                return [self.add_file(copied)]
        try:
            grabbed = ImageGrab.grabclipboard()
        except NotImplementedError:
            raise AttachmentError("Pasting an image needs wl-paste or xclip installed.") from None
        except Exception as e:
            get_logger().warning(
                f"attachments: the clipboard could not be read ({type(e).__name__})"
            )
            get_logger().debug(f"attachments: the clipboard could not be read, detail: {e}")
            raise AttachmentError("The clipboard could not be read.") from None
        if isinstance(grabbed, Image.Image):
            return [self.add_image(grabbed)]
        if isinstance(grabbed, list):
            files = [Path(name) for name in grabbed if _image_format(Path(name)) is not None]
            if files:
                return [self.add_file(path) for path in files]
        raise AttachmentError("No image in the clipboard.")

    def expand(self, text: str) -> str:
        """``text`` with each chip of an attached image replaced by what the model
        needs to look at it. A chip deleted from the draft is not sent; one that names
        no attachment (typed by hand) is left as written."""

        def named(match: re.Match[str]) -> str:
            path = self._paths.get(int(match.group(1)))
            if path is None:
                return match.group(0)
            return (
                f"[Image #{match.group(1)}, attached by the user: {path}. "
                "Look at it with read_image.]"
            )

        return CHIP.sub(named, text)
