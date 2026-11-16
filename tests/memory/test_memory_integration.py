"""Integration tests for memory manager: remember, update_memory, agent validator, configuration_id."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents.agents import Agent
from domyn.agents.core import BaseAgent, InteractionContext
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse
from domyn.agents.memory.managers import (
    BaseMemoryManager,
    FilesystemMemoryManager,
    Memory,
    MemoryExecutionMode,
)

# ── Fixtures ──────────────────────────────────────────────────


class MockLLMProvider(BaseLLMProvider):
    type: str = "mock"
    name: str = "mock-llm"
    model_name: str = "mock-model"
    supported_structured_outputs: None = None

    async def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        return LLMResponse(content="mock", usage={}, generated_tokens=1, execution_time=0.01)

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs):
        yield


@pytest.fixture
def temp_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def memory_manager(temp_file):
    return FilesystemMemoryManager(file_path=temp_file, execution_mode=MemoryExecutionMode.AGENTIC)


@pytest.fixture
def llm_provider():
    return MockLLMProvider()


def _build_agent(name: str = "test_agent") -> BaseAgent:
    agent = AsyncMock(spec=BaseAgent)
    agent.name = name
    agent.sub_agents = []
    return agent


# ── remember() return value ───────────────────────────────────


class TestRememberReturnValue:
    """Verify remember() always returns str, never None — prevents LLM retry loops."""

    @pytest.mark.asyncio
    async def test_remember_returns_str_on_success(self, memory_manager):
        agent = _build_agent("test_agent")
        result = await memory_manager.remember(
            agent=agent,
            memory_content="User likes cats",
            run_config={"user_id": "u1"},
        )
        assert isinstance(result, str)
        assert result == "Memory stored."

    @pytest.mark.asyncio
    async def test_remember_returns_str_without_user_id(self, memory_manager):
        agent = _build_agent("test_agent")
        result = await memory_manager.remember(
            agent=agent,
            memory_content="User likes cats",
            run_config={},
        )
        assert isinstance(result, str)
        assert "not stored" in result.lower()

    @pytest.mark.asyncio
    async def test_remember_returns_str_with_none_run_config(self, memory_manager):
        agent = _build_agent("test_agent")
        result = await memory_manager.remember(
            agent=agent,
            memory_content="User likes cats",
            run_config=None,
        )
        assert isinstance(result, str)
        assert "not stored" in result.lower()


# ── run_config forwarding via **kwargs ────────────────────────


class TestRunConfigForwarding:
    """Verify run_config is passed through to write_single_memory as a kwarg."""

    @pytest.mark.asyncio
    async def test_remember_passes_run_config_to_write(self, memory_manager):
        agent = _build_agent("test_agent")
        with patch(
            "domyn.agents.memory.managers.local.FilesystemMemoryManager.write_single_memory",
            new_callable=AsyncMock,
        ) as mock_write:
            await memory_manager.remember(
                agent=agent,
                memory_content="User is from Rome",
                run_config={"user_id": "u1", "config_id": "cfg-abc-123"},
            )
            mock_write.assert_called_once_with(
                user_id="u1",
                agent=agent,
                memory_content="User is from Rome",
                run_config={"user_id": "u1", "config_id": "cfg-abc-123"},
            )

    @pytest.mark.asyncio
    async def test_remember_stores_memory_without_platform_fields(self, memory_manager):
        agent = _build_agent("test_agent")
        await memory_manager.remember(
            agent=agent,
            memory_content="User is from Rome",
            run_config={"user_id": "u1", "config_id": "cfg-abc-123"},
        )
        memories = await memory_manager.recall_memories("u1")
        assert len(memories) == 1
        assert not hasattr(memories[0], "configuration_id")

    @pytest.mark.asyncio
    async def test_write_single_memory_accepts_kwargs(self, memory_manager):
        agent = _build_agent("agent")
        await memory_manager.write_single_memory(
            user_id="u1",
            agent=agent,
            memory_content="test",
            run_config={"config_id": "cfg-xyz"},
        )
        memories = await memory_manager.recall_memories("u1")
        assert len(memories) == 1
        assert memories[0].memory == "test"


# ── Agent validator (memory_manager from dict) ────────────────


class TestAgentMemoryManagerValidator:
    """Verify field_validator deserializes memory_manager from dict."""

    def _make_agent(self, memory_manager_value):
        return Agent(
            name="TestAgent",
            description="test",
            instruction="test instruction",
            llm_provider=MockLLMProvider(),
            memory_manager=memory_manager_value,
        )

    def test_from_dict_filesystem(self, temp_file):
        agent = self._make_agent({"type": "filesystem", "file_path": temp_file})
        assert isinstance(agent.memory_manager, FilesystemMemoryManager)
        assert agent.memory_manager.file_path == temp_file

    def test_from_none(self):
        agent = self._make_agent(None)
        assert agent.memory_manager is None

    def test_from_instance(self, temp_file):
        mm = FilesystemMemoryManager(file_path=temp_file)
        agent = self._make_agent(mm)
        assert agent.memory_manager is mm

    def test_unknown_type_returns_none(self):
        agent = self._make_agent({"type": "nonexistent_backend"})
        assert agent.memory_manager is None

    def test_missing_type_returns_none(self):
        agent = self._make_agent({"file_path": "/tmp/x.jsonl"})
        assert agent.memory_manager is None

    def test_agentic_mode_adds_remember_tool(self, temp_file):
        agent = self._make_agent(
            {
                "type": "filesystem",
                "file_path": temp_file,
                "execution_mode": "agentic",
            }
        )
        tool_names = [t.name for t in agent.tools]
        assert "remember" in tool_names

    def test_automatic_mode_adds_post_hook(self, temp_file):
        agent = self._make_agent(
            {
                "type": "filesystem",
                "file_path": temp_file,
                "execution_mode": "automatic",
            }
        )
        hook_names = [h.name for h in agent.post_hooks]
        assert "update_memory" in hook_names

    def test_no_memory_manager_no_tools_or_hooks(self):
        agent = self._make_agent(None)
        tool_names = [t.name for t in agent.tools]
        hook_names = [h.name for h in agent.post_hooks]
        assert "remember" not in tool_names
        assert "update_memory" not in hook_names


# ── update_memory flow (AUTOMATIC mode) ──────────────────────


class TestUpdateMemoryFlow:
    """Verify the AUTOMATIC mode: build_memory_content → write_single_memory."""

    def _make_context(self, user_id="u1", run_config=None):
        user_event = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="My name is John and I live in Milan")],
        )
        return InteractionContext(
            conversation_id="conv-1",
            user_id=user_id,
            agent_input=user_event,
            conversation_history=[user_event],
            run_config=run_config or {"config_id": "cfg-1"},
        )

    def _make_agent_mock(self, llm_response_text):
        agent = AsyncMock()
        agent.name = "TestAgent"
        agent.llm_provider.generate.return_value = LLMResponse(
            content=llm_response_text,
            usage={},
            generated_tokens=10,
            execution_time=0.01,
        )
        return agent

    @pytest.mark.asyncio
    async def test_update_memory_stores_when_yes(self, memory_manager):
        agent = self._make_agent_mock(
            "StoreMemory: yes\nMemoryContent: User's name is John, lives in Milan"
        )
        context = self._make_context()

        await memory_manager.update_memory(agent, context)

        memories = await memory_manager.recall_memories("u1")
        assert len(memories) == 1
        assert "John" in memories[0].memory

    @pytest.mark.asyncio
    async def test_the_turn_that_just_ended_is_read(self, memory_manager):
        # When the post-agent hook runs, the turn is in `interaction_history` and a
        # first turn's `conversation_history` is empty: the LLM must still see it.
        user_event = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="I am Pippo, a comedian in Castelfranco Veneto")],
        )
        context = InteractionContext(
            conversation_id="conv-1",
            user_id="u1",
            agent_input=user_event,
            conversation_history=[],
            interaction_history=[user_event],
        )
        agent = self._make_agent_mock("StoreMemory: no")

        await memory_manager.update_memory(agent, context)

        sent = agent.llm_provider.generate.call_args.kwargs["messages"][1].content
        assert "Castelfranco Veneto" in sent

    @pytest.mark.asyncio
    async def test_an_event_in_both_histories_is_sent_once(self, memory_manager):
        context = self._make_context()
        context.interaction_history = list(context.conversation_history)
        agent = self._make_agent_mock("StoreMemory: no")

        await memory_manager.update_memory(agent, context)

        sent = agent.llm_provider.generate.call_args.kwargs["messages"][1].content
        assert sent.count("My name is John") == 1

    @pytest.mark.asyncio
    async def test_the_writer_after_a_decision_reads_the_turn_too(self, memory_manager):
        # Asked after a decision model's yes, it is shown the same conversation as
        # the writer that decides itself: on a first turn, the interaction history.
        user_event = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="I am Pippo, a comedian in Castelfranco Veneto")],
        )
        context = InteractionContext(
            conversation_id="conv-1",
            user_id="u1",
            agent_input=user_event,
            conversation_history=[],
            interaction_history=[user_event, user_event],
        )
        agent = self._make_agent_mock("NONE")

        await memory_manager.write_memory_content(agent, context, ("Pippo", ""))

        sent = agent.llm_provider.generate.call_args.kwargs["messages"][1].content
        conversation = sent.split("Latest exchange:")[0]
        assert conversation.count("Castelfranco Veneto") == 1

    @pytest.mark.asyncio
    async def test_update_memory_skips_when_no(self, memory_manager):
        agent = self._make_agent_mock("StoreMemory: no\nMemoryContent: nothing relevant")
        context = self._make_context()

        await memory_manager.update_memory(agent, context)

        memories = await memory_manager.recall_memories("u1")
        assert len(memories) == 0

    @pytest.mark.asyncio
    async def test_update_memory_tolerates_no_space_after_colon(self, memory_manager):
        agent = self._make_agent_mock("StoreMemory:yes\nMemoryContent:User prefers dark mode")
        context = self._make_context()

        await memory_manager.update_memory(agent, context)

        memories = await memory_manager.recall_memories("u1")
        assert len(memories) == 1
        assert "dark mode" in memories[0].memory

    @pytest.mark.asyncio
    async def test_update_memory_case_insensitive(self, memory_manager):
        agent = self._make_agent_mock("storememory: YES\nmemorycontent: User is left-handed")
        context = self._make_context()

        await memory_manager.update_memory(agent, context)

        memories = await memory_manager.recall_memories("u1")
        assert len(memories) == 1

    @pytest.mark.asyncio
    async def test_update_memory_garbage_response(self, memory_manager):
        agent = self._make_agent_mock("I don't understand the question")
        context = self._make_context()

        await memory_manager.update_memory(agent, context)

        memories = await memory_manager.recall_memories("u1")
        assert len(memories) == 0

    @pytest.mark.asyncio
    async def test_update_memory_passes_run_config_as_kwarg(self, memory_manager):
        agent = self._make_agent_mock("StoreMemory: yes\nMemoryContent: User likes jazz")
        context = self._make_context(run_config={"config_id": "cfg-jazz-42"})

        with patch(
            "domyn.agents.memory.managers.local.FilesystemMemoryManager.write_single_memory",
            new_callable=AsyncMock,
        ) as mock_write:
            await memory_manager.update_memory(agent, context)
            mock_write.assert_called_once()
            _, kwargs = mock_write.call_args
            assert kwargs["run_config"] == {"config_id": "cfg-jazz-42"}

    @pytest.mark.asyncio
    async def test_update_memory_none_run_config(self, memory_manager):
        """update_memory should not crash when run_config is None."""
        agent = self._make_agent_mock("StoreMemory: yes\nMemoryContent: User likes tea")
        context = self._make_context(run_config={})
        context.run_config = None

        await memory_manager.update_memory(agent, context)

        memories = await memory_manager.recall_memories("u1")
        assert len(memories) == 1


# ── The AUTOMATIC hook runs the update in the background ─────


class TestUpdateInTheBackground:
    """The hook returns before the update ends; closing waits for what is left."""

    def _context(self):
        user_event = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="I like tea")],
        )
        return InteractionContext(
            conversation_id="conv-1",
            user_id="u1",
            agent_input=user_event,
            conversation_history=[user_event],
        )

    def _blocked_update(self, memory_manager):
        """An update that waits for `release`, and records that it finished."""
        release = asyncio.Event()
        finished = []

        async def update_memory(agent, context):
            await release.wait()
            finished.append(context.user_id)
            return "User likes tea"

        object.__setattr__(memory_manager, "update_memory", update_memory)
        return release, finished

    @pytest.mark.asyncio
    async def test_the_hook_returns_before_the_update_ends(self, memory_manager):
        release, finished = self._blocked_update(memory_manager)

        result = await memory_manager.start_memory_update(_build_agent(), self._context())

        assert result is None
        assert finished == []
        release.set()
        await memory_manager.wait_for_updates()
        assert finished == ["u1"]

    @pytest.mark.asyncio
    async def test_the_update_stores_the_fact(self, memory_manager):
        agent = AsyncMock()
        agent.name = "TestAgent"
        agent.llm_provider.generate.return_value = LLMResponse(
            content="StoreMemory: yes\nMemoryContent: User likes tea",
            usage={},
            generated_tokens=10,
            execution_time=0.01,
        )

        await memory_manager.start_memory_update(agent, self._context())
        await memory_manager.wait_for_updates()

        memories = await memory_manager.recall_memories("u1")
        assert [m.memory for m in memories] == ["User likes tea"]

    @pytest.mark.asyncio
    async def test_an_update_still_running_at_the_limit_is_dropped(self, memory_manager):
        release, finished = self._blocked_update(memory_manager)
        await memory_manager.start_memory_update(_build_agent(), self._context())

        await memory_manager.wait_for_updates(timeout=0.01)

        release.set()
        await asyncio.sleep(0)
        assert finished == []
        assert not memory_manager._updates

    @pytest.mark.asyncio
    async def test_closing_the_agent_waits_for_the_update(self, temp_file, llm_provider):
        agent = Agent(
            name="Assistant",
            description="d",
            instruction="i",
            llm_provider=llm_provider,
            memory_manager={
                "type": "filesystem",
                "file_path": temp_file,
                "execution_mode": "automatic",
            },
        )
        release, finished = self._blocked_update(agent.memory_manager)
        hook = next(h for h in agent.post_hooks if h.name == "update_memory")
        await hook._actual_callable(agent=agent, context=self._context())

        asyncio.get_running_loop().call_soon(release.set)
        await agent.close()

        assert finished == ["u1"]
