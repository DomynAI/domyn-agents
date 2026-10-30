"""The ``Transcriber`` interface, and turning a configuration into one.

A transcriber registers itself by its ``type`` tag, like skill and instruction
sources, so one can be named in YAML or in ``~/.domyn/config.yaml``
(``{"type": "local"}``, ``{"type": "openai", "url": ...}``) and a new engine is
a subclass, not a change here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from domyn.agents.audio.models import Transcript

AudioSource = str | Path | bytes
"""A path to an audio file, or its bytes."""


class Transcriber(BaseModel, ABC):
    """Speech to text."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _registry: ClassVar[dict[str, type[Transcriber]]] = {}

    type: str
    language: str | None = Field(
        default=None,
        description="ISO 639-1 code of the speech (`it`, `en`); None detects it. "
        "Engines that always detect it themselves ignore it.",
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        # After the model is built, so `model_fields` is populated: see
        # SkillSource.__pydantic_init_subclass__.
        super().__pydantic_init_subclass__(**kwargs)
        field = cls.model_fields.get("type")
        tag = getattr(field, "default", None)
        if isinstance(tag, str) and tag:
            Transcriber._registry[tag] = cls

    @abstractmethod
    async def transcribe(self, audio: AudioSource, *, filename: str | None = None) -> Transcript:
        """The text of ``audio``.

        ``filename`` names bytes passed in place of a path: its extension says
        what format they are in. Raises ``TranscriptionError`` when the audio
        cannot be transcribed.
        """

    def is_ready(self) -> bool:
        """Whether ``transcribe`` can start without a download first."""
        return True

    async def prepare(self, on_download: Callable[[str], None] | None = None) -> None:
        """Do now what the first ``transcribe`` would otherwise do first.

        For an engine that downloads a model, the download: ``on_download`` is
        called with a line saying what is downloaded and where, before it starts,
        and only when there is something to download. Nothing for the others.
        """

    def describe(self) -> str:
        return self.type


def resolve_transcriber(value: Transcriber | dict[str, Any]) -> Transcriber:
    """A transcriber from its configuration, ``{"type": ..., ...}``.

    An unknown ``type`` raises rather than falling back to a default: a typo
    would otherwise send audio somewhere other than where it was configured to go.
    """
    if isinstance(value, Transcriber):
        return value
    tag = value.get("type")
    if not tag:
        raise ValueError("transcriber configuration is missing 'type'")
    subclass = Transcriber._registry.get(tag)
    if subclass is None:
        known = ", ".join(sorted(Transcriber._registry)) or "none registered"
        raise ValueError(f"unknown transcriber type '{tag}' (known: {known})")
    return subclass.model_validate(value)
