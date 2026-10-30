"""A stand-in for the platform (the ``platform`` fixture), so tests run without ``domyn-platform-sdk``."""

from typing import Any

import pytest

from domyn.agents.integrations import domyn_platform_sdk


class FakePlatform:
    """Answers every platform tool listing, and records what was asked."""

    def __init__(self) -> None:
        self.tools: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.calls: list[dict[str, Any]] = []

    def serve(self, tools: list[dict[str, Any]]) -> None:
        self.tools, self.error = tools, None

    def fail(self, error: Exception | None = None) -> None:
        self.error = error or RuntimeError("platform unavailable")

    @property
    def called(self) -> bool:
        return bool(self.calls)


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch) -> FakePlatform:
    fake = FakePlatform()

    async def list_tools(
        self: domyn_platform_sdk.SdkToolCatalog, *, agent_name: str | None = None
    ) -> list[dict[str, Any]]:
        fake.calls.append({"settings": self._settings, "agent_name": agent_name})
        if fake.error is not None:
            raise fake.error
        return list(fake.tools)

    monkeypatch.setattr(domyn_platform_sdk.SdkToolCatalog, "list_tools", list_tools)
    return fake
