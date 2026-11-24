"""Reading an image, by asking the agent's own model about it.

An image never enters the conversation. ``read_image`` sends it, with a
question, to the provider the agent itself uses, in a request of its own, and
what comes back as the tool's result is the answer in text. The history
therefore stays text: nothing to store, replay or compact differently, and
nothing a later request has to pay for again.

The tool exists only for a model that reads images (``BaseLLMProvider.vision``),
and is bound to that provider and to a workspace when it is built, like every
built-in: neither is something the model can pass.

The image is sent as the file is when every provider takes it that way: PNG,
JPEG, GIF or WebP, within :data:`MAX_IMAGE_EDGE` and :data:`MAX_SENT_BYTES`.
Anything else Pillow opens is scaled down and sent as JPEG.
"""

# No `from __future__ import annotations` here: FunctionTool reads the signature
# of the tool at construction time and cannot resolve string annotations.

import asyncio
import base64
import io
from typing import TYPE_CHECKING, Any

from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.disk import LOCAL_DISK, Disk, LocalDisk
from domyn.agents.tools.builtin.workspace import Workspace, WorkspaceError

if TYPE_CHECKING:
    from domyn.agents.llm.base import BaseLLMProvider
    from domyn.agents.tools.builtin.backends.base import CommandBackend

from domyn.agents.tools.builtin.files import (
    IMAGE_OPERATION,
    TOOL_BODY_SECONDS,
    answer_of,
    host_path,
    on_this_machine,
    request_text,
)

# Longest edge sent, in pixels. Providers scale a larger image down themselves
# (or refuse it), so the pixels past this are paid for in upload and read by nobody.
MAX_IMAGE_EDGE = 2048

# Largest image sent as the file is. Base64 adds a third, and the lowest limit
# among the providers is Anthropic's 5 MB for the encoded image.
MAX_SENT_BYTES = 3_500_000

# The formats every provider takes, by the name Pillow gives them.
_SENT_AS_IS = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}

# The model answering sees the image and the question, and nothing of the
# conversation they come from.
_FRAMING = (
    "Answer from what is visible in the image. Copy any text in it exactly as written. "
    "If what is asked about is not in the image, say so instead of guessing."
)


class ImageError(WorkspaceError):
    """An image could not be read. Raised, not returned, like ``WorkspaceError``: the
    message is written for the model. A ``WorkspaceError`` so that, raised in a
    container, it comes back as a refusal with its message unchanged (``files.serve``)."""


def load_image(workspace: Workspace, path: str, disk: Disk = LOCAL_DISK) -> tuple[bytes, str]:
    """The image at ``path`` as it goes to the provider: its bytes and their media type.

    Synchronous: decoding is the slow part, so it runs in a thread here, or in the process
    the tool host starts in a container (``files.run_file_operation``).
    """
    resolved = disk.resolve(workspace, path, must_exist=True)
    if disk.is_dir(resolved):
        raise WorkspaceError(f"'{path}' is a directory. Use list_dir for that.")
    # Read once from another machine; here, opened by its path as it always was.
    held = None if isinstance(disk, LocalDisk) else disk.read_bytes(resolved)
    try:
        with Image.open(resolved if held is None else io.BytesIO(held)) as image:
            media_type = _SENT_AS_IS.get(image.format or "")
            upright = image.getexif().get(ExifTags.Base.Orientation, 1) == 1
            if (
                media_type
                and upright
                and max(image.size) <= MAX_IMAGE_EDGE
                and disk.size(resolved) <= MAX_SENT_BYTES
            ):
                return (held if held is not None else disk.read_bytes(resolved)), media_type
            trasposed_image = ImageOps.exif_transpose(image)
            trasposed_image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
            # JPEG has no transparency, and a transparent area converted as it is turns
            # black: a diagram with black lines would be sent as a black rectangle.
            layer = trasposed_image.convert("RGBA")
            flat = Image.new("RGB", layer.size, "white")
            flat.paste(layer, mask=layer.getchannel("A"))
            buffer = io.BytesIO()
            flat.save(buffer, format="JPEG", quality=85)
    except UnidentifiedImageError:
        raise ImageError(
            f"'{path}' is not an image this tool can open. Use read_file for text and documents."
        ) from None
    except Image.DecompressionBombError:
        raise ImageError(
            f"'{path}' has too many pixels to open. Retrying will not help; try a smaller version."
        ) from None
    except OSError as e:
        get_logger().warning(f"read_image: could not decode an image ({type(e).__name__})")
        get_logger().debug(f"read_image: could not decode '{resolved}', detail: {e}")
        raise ImageError(
            f"'{path}' could not be decoded: the file is damaged or cut short."
        ) from None

    return buffer.getvalue(), "image/jpeg"


def image_tools(
    workspace: Workspace, provider: "BaseLLMProvider", backend: "CommandBackend | None" = None
) -> list[Any]:
    """``read_image``, bound to one workspace and to the model that answers."""
    from domyn.agents.tools.function_tool import FunctionTool

    async def read_image(path: str, query: str, conversation_id: str = "") -> str:
        """Look at an image file and answer a question about it.

        The image is not shown to you: a separate request looks at it, and what
        comes back is the answer in text. Ask for everything you need, such as the
        text in it, the values in a chart or the layout of a screen, and call again
        with another question for more detail. Reads PNG, JPEG, GIF, WebP and the
        other common image formats. Not for PDF or other documents: use read_file.

        Args:
            path: Path to the image, relative to the workspace root, or the path
                of an image the user attached, as the message gives it.
            query: What to find out about the image, for example "Transcribe all
                the text" or "What does the error dialog say?".
        """

        if backend is not None:
            # Where an "auto" backend decides between Docker and this machine
            await backend.start()
        where = backend.tool_host if backend is not None else None
        remote = backend.file_disk() if backend is not None else None
        if (backend is None) or (where is None):  # Local execution
            # The images the user attached to this conversation are in its state
            # folder, which is no root of the workspace: they are added for this call
            # alone, and only this conversation's. Never through a link, which would
            # make a root of wherever it points. In a container nothing is needed:
            # the conversation's folder is mounted there, and `/` is a root.
            reach = workspace
            if conversation_id:
                from domyn.agents.tools.builtin.session import (
                    attachments_dir,
                    default_state_dir,
                    linked_below,
                    state_base,
                )

                # The attachments are on this machine whatever the backend: with the
                # files on another, its state folder is there, not here.
                state_dir = (
                    backend.state_dir
                    if backend is not None and remote is None
                    else default_state_dir()
                )
                attached = attachments_dir(workspace.root, state_dir, conversation_id)
                start = state_base(workspace.root, state_dir)
                if attached.is_dir() and linked_below(start, attached) is None:
                    reach = Workspace.at(workspace.root, also=(*workspace.extra_roots, attached))
            if remote is not None and not on_this_machine(path, reach):
                space, disk = remote
                image, media_type = await asyncio.to_thread(
                    load_image, workspace=space, path=path, disk=disk
                )
            else:
                image, media_type = await asyncio.to_thread(load_image, workspace=reach, path=path)
        else:  # Remote execution
            remote_path = str(host_path(path, workspace))
            result = await backend.run(
                where.command,
                timeout=TOOL_BODY_SECONDS,
                conversation_id=conversation_id,
                stdin=request_text(
                    IMAGE_OPERATION,
                    {"path": remote_path},
                    root=workspace.root,
                    extra_roots=(*workspace.extra_roots, *where.roots),
                ),
            )
            header, data = answer_of(result, "read_image").split(",", 1)
            media_type = header.removeprefix("data:").removesuffix(";base64")
            image = base64.b64decode(data)

        question = query.strip() or "Describe the image in detail."

        from domyn.agents.llm.base import image_message

        try:
            response = await provider.generate(
                [image_message(f"{question}\n\n{_FRAMING}", image, media_type)]
            )
        except Exception as e:
            get_logger().warning(f"read_image: the model call failed ({type(e).__name__})")
            get_logger().debug(f"read_image: the model call failed, detail: {e}")
            status = getattr(getattr(e, "response", None), "status_code", None)
            if isinstance(status, int) and 400 <= status < 500:
                raise ImageError(
                    f"The model refused '{path}' (HTTP {status}). Retrying will not help."
                ) from None
            raise ImageError(
                f"The model could not be asked about '{path}' ({type(e).__name__})."
            ) from None

        answer = response.content.strip()
        if not answer:
            raise ImageError(f"The model gave no answer about '{path}'.")
        if response.truncated:
            answer += "\n\n[The answer was cut at the model's output limit.]"
        return answer

    return [FunctionTool(func=read_image)]
