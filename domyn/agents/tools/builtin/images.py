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
import io
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from domyn.agents.llm.base import BaseLLMProvider, image_message
from domyn.agents.logger import get_logger
from domyn.agents.tools.builtin.workspace import Workspace, WorkspaceError

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


class ImageError(Exception):
    """An image could not be read. Raised, not returned, like ``WorkspaceError``:
    the message is written for the model."""


def _as_sent(path: Path) -> tuple[bytes, str]:
    """The image as it goes to the provider: its bytes and their media type.

    A function of its own because decoding is the slow part and runs in a thread.
    """
    with Image.open(path) as image:
        media_type = _SENT_AS_IS.get(image.format or "")
        if (
            media_type
            and max(image.size) <= MAX_IMAGE_EDGE
            and path.stat().st_size <= MAX_SENT_BYTES
        ):
            return path.read_bytes(), media_type
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
        # JPEG has no transparency, and a transparent area converted as it is turns
        # black: a diagram with black lines would be sent as a black rectangle.
        layer = image.convert("RGBA")
        flat = Image.new("RGB", layer.size, "white")
        flat.paste(layer, mask=layer.getchannel("A"))
        buffer = io.BytesIO()
        flat.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue(), "image/jpeg"


def image_tools(workspace: Workspace, provider: BaseLLMProvider) -> list[Any]:
    """``read_image``, bound to one workspace and to the model that answers."""
    from domyn.agents.tools.function_tool import FunctionTool

    async def read_image(path: str, query: str) -> str:
        """Look at an image file and answer a question about it.

        The image is not shown to you: a separate request looks at it, and what
        comes back is the answer in text. Ask for everything you need, such as the
        text in it, the values in a chart or the layout of a screen, and call again
        with another question for more detail. Reads PNG, JPEG, GIF, WebP and the
        other common image formats. Not for PDF or other documents: use read_file.

        Args:
            path: Path to the image, relative to the workspace root.
            query: What to find out about the image, for example "Transcribe all
                the text" or "What does the error dialog say?".
        """
        resolved = workspace.resolve(path, must_exist=True)
        if resolved.is_dir():
            raise WorkspaceError(f"'{path}' is a directory. Use list_dir for that.")
        try:
            image, media_type = await asyncio.to_thread(_as_sent, resolved)  # Review from here
        except UnidentifiedImageError:
            raise ImageError(
                f"'{path}' is not an image this tool can open. "
                "Use read_file for text and documents."
            ) from None
        except Image.DecompressionBombError:
            raise ImageError(
                f"'{path}' has too many pixels to open. Retrying will not help; "
                "try a smaller version."
            ) from None
        except OSError as e:
            get_logger().warning(f"read_image: could not decode an image ({type(e).__name__})")
            get_logger().debug(f"read_image: could not decode '{resolved}', detail: {e}")
            raise ImageError(
                f"'{path}' could not be decoded: the file is damaged or cut short."
            ) from None

        question = query.strip() or "Describe the image in detail."
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
