"""Skills are loaded once per turn, by the runner, for every agent in the tree.

The bug this replaces: resolution lived in the planner and therefore ran once per
*agent entry*. A turn that goes A → B → A swept A's source twice, and a local
source re-reads and re-parses every skill file on each sweep.

These are the guards that the load happens once, covers agents the turn has not
reached yet, survives a source being down, and never changes how an agent that
declares no skills is called.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import Field

from domyn.agents.agents import Agent
from domyn.agents.conversation_manager import ConversationInfo, InMemoryConversationManager
from domyn.agents.core import (
    AgentAction,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.runner import Runner, RunnerState
from domyn.agents.skills import (
    LoadedSkills,
    LocalDirSkillSource,
    Selection,
    SkillBudget,
    SkillContent,
    SkillDescriptor,
    SkillIndex,
    SkillNotFoundError,
    SkillQuery,
    SkillRef,
    SkillSelector,
    SkillSource,
    SkillsRuntime,
)

APP = "TestApplication"
USER = "test_user"


class CountingSource(SkillSource):
    """Counts loads. One per turn is the contract; more is the regression."""

    type: str = "counting_load"
    calls: int = 0
    seen_tasks: list[str | None] = Field(default_factory=list)

    async def load(self, query: SkillQuery) -> LoadedSkills:
        self.calls += 1
        self.seen_tasks.append(query.task)
        return LoadedSkills(
            index=SkillIndex(skills=[SkillDescriptor(name="sql-style", description="how we sql")]),
            bodies={"sql-style": SkillContent(name="sql-style", body="use CTEs")},
        )

    async def index(self, query: SkillQuery) -> SkillIndex:
        return (await self.load(query)).index

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        raise SkillNotFoundError(ref.name)


class DeadSource(SkillSource):
    """A source that is down, and counts how often it is asked."""

    type: str = "dead_load"
    calls: int = 0

    async def load(self, query: SkillQuery) -> LoadedSkills:
        self.calls += 1
        raise OSError("skills backend unreachable")

    async def index(self, query: SkillQuery) -> SkillIndex:
        return (await self.load(query)).index

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        raise SkillNotFoundError(ref.name)


class UnavailableRequiredSource(SkillSource):
    """Advertises a ``required`` skill whose requirements are unmet.

    Unmet, not unreadable: the descriptor exists and says ``required``, which is
    the only state in which enforcement can be known at all. ``enforcement`` is
    declared *inside* the file, so a file that cannot be read cannot be known to
    have declared it — that is ``DeadSource``, and it must still degrade.
    """

    type: str = "unavailable_required_load"
    enforcement: str = "required"

    async def load(self, query: SkillQuery) -> LoadedSkills:
        return LoadedSkills(
            index=SkillIndex(
                skills=[
                    SkillDescriptor(
                        name="refund-policy",
                        description="mandatory",
                        enforcement=self.enforcement,
                        available=False,
                        unavailable_reason="binary 'nope-not-real'",
                    )
                ]
            )
        )

    async def index(self, query: SkillQuery) -> SkillIndex:
        return (await self.load(query)).index

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        raise SkillNotFoundError(ref.name)


class InlineFetchSource(SkillSource):
    """Index-only up front, like any source whose bodies cost a round trip."""

    type: str = "inline_fetch_load"
    fetches: int = 0

    async def load(self, query: SkillQuery) -> LoadedSkills:
        return LoadedSkills(
            index=SkillIndex(skills=[SkillDescriptor(name="sql-style", description="how we sql")])
        )

    async def index(self, query: SkillQuery) -> SkillIndex:
        return (await self.load(query)).index

    async def fetch(self, ref: SkillRef, query: SkillQuery) -> SkillContent:
        self.fetches += 1
        if ref.name != "sql-style":
            raise SkillNotFoundError(ref.name)
        return SkillContent(name=ref.name, body="use CTEs")


def _provider() -> Any:
    from domyn.agents.llm import CustomLLMProvider

    return CustomLLMProvider(model_name="m", url="http://x")


def _agent(name: str, *, skills: Any = None, sub_agents: list[Any] | None = None) -> Agent:
    return Agent(
        name=name,
        description=f"{name} description",
        instruction="i",
        llm_provider=_provider(),
        skills=skills,
        sub_agents=sub_agents or [],
    )


def _scripted(batches: list[list[BaseEvent]], sink: list[dict[str, Any]]):
    """Replace ``_execute``: yield one batch per entry, recording the kwargs.

    Recording the kwargs is what makes the once-per-turn assertion real — a second
    entry that carried a *different* LoadedSkills object would mean a reload.
    """
    remaining = list(batches)

    async def _gen(context, **kwargs):
        sink.append(kwargs)
        for event in remaining.pop(0) if remaining else []:
            yield event

    return _gen


def _user_input(text: str = "how do I write this query") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text=text)],
        turn_id="turn_1",
    )


async def _run(runner: Runner, root: Agent, event: BaseEvent, conversation_id: str) -> list:
    return [
        e
        async for e in runner.run(
            application_name=APP,
            user_id=USER,
            conversation_id=conversation_id,
            root=root,
            user_input=event,
        )
    ]


class TestLoadedOncePerTurn:
    @pytest.mark.asyncio
    async def test_a_handoff_and_back_loads_once(self, runner: Runner) -> None:
        """A → B → A: one load for A, not one per entry."""
        source = CountingSource()
        sub = _agent("SubAgent")
        main = _agent("MainAgent", skills=SkillsRuntime(source=source), sub_agents=[sub])

        main_calls: list[dict[str, Any]] = []
        main._execute = _scripted(
            [
                [
                    BaseEvent(
                        event_type=ExecutionEventType.AGENT_CALL,
                        author="MainAgent",
                        action=AgentAction(name="SubAgent", parameters={"task": "look it up"}),
                    )
                ],
                [
                    BaseEvent(
                        event_type=ExecutionEventType.RESPONSE,
                        author="MainAgent",
                        content=[Part(text="all done")],
                    )
                ],
            ],
            main_calls,
        )
        sub._execute = _scripted(
            [
                [
                    BaseEvent(
                        event_type=ExecutionEventType.RESPONSE,
                        author="SubAgent",
                        content=[Part(text="looked up")],
                    )
                ]
            ],
            [],
        )

        await _run(runner, main, _user_input(), "skills_once_per_turn")

        assert len(main_calls) == 2, "the turn should have entered MainAgent twice"
        assert source.calls == 1, (
            f"loaded {source.calls}x for one turn — the load moved back into the "
            "agent loop instead of staying in run()"
        )
        first, second = main_calls
        assert first["skills"] is second["skills"], (
            "the second entry got a different LoadedSkills object — it reloaded"
        )

    @pytest.mark.asyncio
    async def test_an_agent_the_turn_never_reaches_is_still_loaded(self, runner: Runner) -> None:
        """Eager, because which agents a turn will reach is not knowable up front.

        Discovering it at entry is exactly the per-entry load this removes.
        """
        source = CountingSource()
        sub = _agent("SubAgent", skills=SkillsRuntime(source=source))
        main = _agent("MainAgent", sub_agents=[sub])
        main._execute = _scripted(
            [
                [
                    BaseEvent(
                        event_type=ExecutionEventType.RESPONSE,
                        author="MainAgent",
                        content=[Part(text="answered without delegating")],
                    )
                ]
            ],
            [],
        )

        await _run(runner, main, _user_input(), "skills_eager")

        assert source.calls == 1


class TestARequiredSkillBlocksTheTurn:
    """``enforcement: required`` with unmet requirements refuses the whole turn.

    The whole tree, not only the agent about to run: which agents a turn reaches
    is not knowable up front, so a guardrail checked on entry is one the turn can
    route around by taking another path. The offending agent here is a sub-agent
    the scripted root never even hands off to.
    """

    @staticmethod
    def _tree(
        enforcement: str = "required", budget: SkillBudget | None = None
    ) -> tuple[Agent, Agent, list[dict[str, Any]]]:
        sub = _agent(
            "SubAgent",
            skills=SkillsRuntime(
                source=UnavailableRequiredSource(enforcement=enforcement),
                **({"budget": budget} if budget is not None else {}),
            ),
        )
        main = _agent("MainAgent", sub_agents=[sub])
        main_calls: list[dict[str, Any]] = []
        main._execute = _scripted(
            [
                [
                    BaseEvent(
                        event_type=ExecutionEventType.RESPONSE,
                        author="MainAgent",
                        content=[Part(text="done")],
                    )
                ]
            ],
            main_calls,
        )
        return main, sub, main_calls

    @pytest.mark.asyncio
    async def test_an_unavailable_required_skill_on_a_subagent_blocks_the_turn(
        self, runner: Runner
    ) -> None:
        """The headline: no agent runs, and the error says enough to fix it.

        ``MainAgent``'s own skills are fine and the turn would never have reached
        ``SubAgent``. It is still refused, before any model is called.
        """
        main, _, main_calls = self._tree()

        events = await _run(runner, main, _user_input(), "skills_required_block")

        assert main_calls == [], "the root ran despite a guardrail it could not satisfy"
        errors = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
        assert len(errors) == 1
        assert errors[0].error_code == "RequiredSkillUnavailable"
        assert errors[0].author == "MainAgent"
        # All three, because an operator has to fix this from the message alone:
        # which agent, which skill, and why it is unavailable.
        message = errors[0].error_message or ""
        for fragment in ("SubAgent", "refund-policy", "nope-not-real"):
            assert fragment in message, f"{fragment!r} missing from {message!r}"
        assert not [e for e in events if e.event_type == ExecutionEventType.RESPONSE]

    @pytest.mark.asyncio
    async def test_a_tight_index_budget_does_not_disarm_the_guardrail(self, runner: Runner) -> None:
        """The index budget must not decide whether enforcement happens.

        ``_load_turn_skills`` checks the index ``load()`` returns, and that index is
        already trimmed to ``max_index_chars``. A guardrail the budget dropped used
        to be a guardrail that silently never fired.
        """
        main, _, main_calls = self._tree(budget=SkillBudget(max_index_chars=1))

        events = await _run(runner, main, _user_input(), "skills_required_tight_budget")

        assert main_calls == [], "the root ran despite a guardrail the budget had hidden"
        errors = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
        assert len(errors) == 1
        assert errors[0].error_code == "RequiredSkillUnavailable"
        assert "refund-policy" in (errors[0].error_message or "")
        assert not [e for e in events if e.event_type == ExecutionEventType.RESPONSE]

    @pytest.mark.asyncio
    async def test_the_refusal_is_persisted_in_history(self, runner: Runner) -> None:
        """The USER_INPUT was stored; a refusal that is not would read as ignored."""
        main, _, _ = self._tree()
        cid = "skills_required_persisted"

        await _run(runner, main, _user_input(), cid)

        conversation = await runner.conversation_manager.get_conversation(APP, USER, cid)
        assert conversation is not None
        assert [
            e.error_code
            for e in conversation.events_history
            if e.event_type == ExecutionEventType.ERROR_OCCURRED
        ] == ["RequiredSkillUnavailable"]

    @pytest.mark.asyncio
    async def test_the_agent_stack_is_emptied_so_the_next_turn_is_not_a_resumption(
        self, runner: Runner
    ) -> None:
        """The entrypoint was pushed before the load; no agent runs to pop it.

        A stale entry makes the next user turn look like a resumption, which
        suppresses its AGENT_START — the bug test_hook_break_stack_cleanup.py
        exists for.
        """
        main, _, _ = self._tree()
        cid = "skills_required_stack"

        await _run(runner, main, _user_input(), cid)

        assert await runner.conversation_manager.get_last_agent_stack(APP, USER, cid) is None

    @pytest.mark.asyncio
    async def test_a_missing_external_file_on_a_subagent_blocks_the_turn(
        self, runner: Runner, tmp_path: Path
    ) -> None:
        """The same guardrail, tripped by a file on disk rather than a fake source."""
        folder = tmp_path / "refund-policy"
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            "---\nname: refund-policy\ndescription: d\nenforcement: required\n"
            "external_files:\n  - rules.py\n---\nbody\n"
        )
        sub = _agent("SubAgent", skills=SkillsRuntime(source=LocalDirSkillSource(paths=[tmp_path])))
        main = _agent("MainAgent", sub_agents=[sub])
        main_calls: list[dict[str, Any]] = []
        main._execute = _scripted([], main_calls)

        events = await _run(runner, main, _user_input(), "skills_required_missing_file")

        assert main_calls == []
        errors = [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
        assert [e.error_code for e in errors] == ["RequiredSkillUnavailable"]
        assert "file 'rules.py' (does not exist)" in (errors[0].error_message or "")

    @pytest.mark.asyncio
    async def test_an_unavailable_advisory_skill_runs_the_turn(self, runner: Runner) -> None:
        """Same source, one word different. Advisory degrades; it never blocks."""
        main, _, main_calls = self._tree(enforcement="advisory")

        events = await _run(runner, main, _user_input(), "skills_advisory_runs")

        assert not [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]
        assert any(e.event_type == ExecutionEventType.RESPONSE for e in events)
        assert len(main_calls) == 1


class TestFailureAndAbsence:
    @pytest.mark.asyncio
    async def test_a_dead_source_degrades_and_is_not_retried(self, runner: Runner) -> None:
        """An advisory skill must never take a turn down, or retry per entry."""
        source = DeadSource()
        sub = _agent("SubAgent")
        main = _agent("MainAgent", skills=SkillsRuntime(source=source), sub_agents=[sub])

        main_calls: list[dict[str, Any]] = []
        main._execute = _scripted(
            [
                [
                    BaseEvent(
                        event_type=ExecutionEventType.AGENT_CALL,
                        author="MainAgent",
                        action=AgentAction(name="SubAgent", parameters={"task": "t"}),
                    )
                ],
                [
                    BaseEvent(
                        event_type=ExecutionEventType.RESPONSE,
                        author="MainAgent",
                        content=[Part(text="done")],
                    )
                ],
            ],
            main_calls,
        )
        sub._execute = _scripted(
            [
                [
                    BaseEvent(
                        event_type=ExecutionEventType.RESPONSE,
                        author="SubAgent",
                        content=[Part(text="ok")],
                    )
                ]
            ],
            [],
        )

        events = await _run(runner, main, _user_input(), "skills_dead_source")

        assert source.calls == 1, "a failed load must not be retried on re-entry"
        assert len(main_calls) == 2
        assert main_calls[0]["skills"] == LoadedSkills(), "a failure should leave an empty load"
        assert any(e.event_type == ExecutionEventType.RESPONSE for e in events)
        # A failed *load* degrades; it never blocks. `enforcement` is declared
        # inside the file, so a source that could not be read never said whether
        # anything was required — the two failures must not converge.
        assert not [e for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED]

    @pytest.mark.asyncio
    async def test_a_strict_inner_execute_is_not_handed_the_kwarg(self, runner: Runner) -> None:
        """Guards the ``Agent.execute`` → ``_execute`` boundary.

        The suite is full of ``_execute`` stubs with a strict ``(context)``
        signature; forwarding ``skills=None`` to them would be a TypeError.
        """
        main = _agent("MainAgent")

        async def strict(context):
            yield BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="MainAgent",
                content=[Part(text="done")],
            )

        main._execute = strict

        events = await _run(runner, main, _user_input(), "skills_no_kwarg")

        assert any(e.event_type == ExecutionEventType.RESPONSE for e in events)

    @pytest.mark.asyncio
    async def test_a_strict_execute_is_not_handed_the_kwarg(self) -> None:
        """Guards the ``runner`` → ``execute`` boundary, which is the other one.

        ``DelegateAgent.execute`` and ``WrappedAgent.execute`` take ``(context)``
        and nothing else — they have no ``skills`` field, so they must be called
        exactly as before. Exercised as the call sites spell it, because an agent
        whose *outer* execute is strict cannot be an ``Agent`` at all and so cannot
        be driven through a real tree.
        """
        runner = Runner(conversation_manager=InMemoryConversationManager())
        entered: list[str] = []

        class StrictAgent:
            name = "Strict"

            async def execute(self, context):  # deliberately no **kwargs
                entered.append("called")
                yield BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="Strict",
                    content=[Part(text="done")],
                )

        agent = StrictAgent()
        state = RunnerState(
            application_name=APP,
            user_id=USER,
            conversation_id="c",
            conversation_info=MagicMock(),
            current_agent=agent,
            available_agents=[agent],
            entrypoint_agent=agent,
        )
        context = InteractionContext(
            conversation_id="c",
            user_id=USER,
            agent_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="q")]
            ),
        )

        assert runner._execute_kwargs(state, agent) == {}

        async for _ in agent.execute(context, **runner._execute_kwargs(state, agent)):
            pass

        assert entered == ["called"]

    @pytest.mark.asyncio
    async def test_the_kwarg_is_handed_over_when_there_is_one(self) -> None:
        """The other half: the helper is not simply always empty."""
        runner = Runner(conversation_manager=InMemoryConversationManager())
        agent = _agent("MainAgent", skills=SkillsRuntime(source=CountingSource()))
        state = RunnerState(
            application_name=APP,
            user_id=USER,
            conversation_id="c",
            conversation_info=MagicMock(),
            current_agent=agent,
            available_agents=[agent],
            entrypoint_agent=agent,
        )
        loaded = LoadedSkills(index=SkillIndex(skills=[]))
        state.skills["MainAgent"] = loaded

        assert runner._execute_kwargs(state, agent) == {"skills": loaded}

    @pytest.mark.asyncio
    async def test_a_compaction_request_loads_nothing(self, runner: Runner) -> None:
        """No agent runs, so no skill I/O — the load sits after that early return."""
        source = CountingSource()
        main = _agent("MainAgent", skills=SkillsRuntime(source=source))

        await _run(
            runner,
            main,
            BaseEvent(
                event_type=ExecutionEventType.COMPACTION_REQUESTED,
                author="user",
                content=[Part(text="/compact")],
                turn_id="turn_1",
            ),
            "skills_compaction",
        )

        assert source.calls == 0


class TestTheIsinstanceGuard:
    @pytest.mark.asyncio
    async def test_a_magicmock_agent_is_skipped(self) -> None:
        """``getattr(agent, "skills", None) is not None`` would NOT be enough.

        BaseAgent declares no ``skills``, and a MagicMock answers every getattr
        with a truthy child mock, which would then blow up on ``.load()``.
        """
        runner = Runner(conversation_manager=InMemoryConversationManager())
        mock_agent = MagicMock()
        mock_agent.name = "Mocky"
        state = RunnerState(
            application_name=APP,
            user_id=USER,
            conversation_id="c",
            conversation_info=MagicMock(),
            current_agent=mock_agent,
            available_agents=[mock_agent],
            entrypoint_agent=mock_agent,
        )
        context = InteractionContext(
            conversation_id="c",
            user_id=USER,
            agent_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="q")]
            ),
        )

        await runner._load_turn_skills(state, context)

        assert state.skills == {}


class TestWhichTaskEachHalfSees:
    @pytest.mark.asyncio
    async def test_the_source_sees_the_turn_and_the_selector_sees_the_entry(
        self, runner: Runner
    ) -> None:
        """The single test that pins the whole split.

        Loading happens once, before any agent runs, so it can only know the text
        the user typed. Selection happens per entry, so a sub-agent's selector
        routes on the task it was actually handed.
        """
        selector_tasks: list[str | None] = []

        class RecordingSelector(SkillSelector):
            type: str = "recording_task"

            async def select(self, index: SkillIndex, query: SkillQuery) -> Selection:
                selector_tasks.append(query.task)
                return Selection(index=list(index.skills), expand=[])

        source = CountingSource()
        sub = _agent("SubAgent", skills=SkillsRuntime(source=source, selector=RecordingSelector()))
        main = _agent("MainAgent", sub_agents=[sub])

        main._execute = _scripted(
            [
                [
                    BaseEvent(
                        event_type=ExecutionEventType.AGENT_CALL,
                        author="MainAgent",
                        action=AgentAction(
                            name="SubAgent", parameters={"task": "check the refund policy"}
                        ),
                    )
                ],
                [
                    BaseEvent(
                        event_type=ExecutionEventType.RESPONSE,
                        author="MainAgent",
                        content=[Part(text="done")],
                    )
                ],
            ],
            [],
        )

        recorded: list[dict[str, Any]] = []

        async def sub_planner(agent, context, **kwargs):
            recorded.append(kwargs)
            yield BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author=agent.name,
                content=[Part(text="looked up")],
            )

        sub.planner = MagicMock()
        sub.planner.execute = sub_planner
        # The runner reads planner.context_manager.prompting.stop_action.name when
        # it converts a sub-agent's RESPONSE into an AGENT_END; a bare mock hands
        # it a MagicMock where a str is required.
        sub.planner.context_manager = None

        await _run(runner, main, _user_input("what is your refund policy?"), "skills_task_split")

        assert source.seen_tasks == ["what is your refund policy?"], (
            "the load ran before any agent, so it can only see the user's text"
        )
        assert selector_tasks == ["check the refund policy"], (
            "the selector must route on the task this agent was handed"
        )
        assert recorded and "sql-style" in recorded[0]["skills_payload"]


class TestTheTurnLoadReachesSkillView:
    """The other half of loading once: the tool call has to spend it.

    Loading once per turn buys nothing if ``skill_view`` then goes back to the
    source for every body the model asks for — which is what it did, because the
    tool is built at agent construction and closes over the runtime alone. The
    runner hands the turn's ``LoadedSkills`` down as a reserved kwarg, the way it
    already does for ``run_config`` and ``conversation_id``.
    """

    @staticmethod
    def _state(agent: Agent, loaded: LoadedSkills | None) -> RunnerState:
        state = RunnerState(
            application_name=APP,
            user_id=USER,
            conversation_id="c1",
            conversation_info=ConversationInfo(
                application_name=APP,
                user_id=USER,
                conversation_id="c1",
                events_history=[],
            ),
            current_agent=agent,
            available_agents=[agent],
            entrypoint_agent=agent,
        )
        if loaded is not None:
            state.skills[agent.name] = loaded
        return state

    @staticmethod
    def _context() -> InteractionContext:
        return InteractionContext(
            conversation_id="c1",
            user_id=USER,
            agent_input=_user_input(),
            state={},
            run_config={},
        )

    @staticmethod
    def _call_event(name: str) -> BaseEvent:
        return BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author="MainAgent",
            content=[],
            action=ToolAction(name="skill_view", parameters={"name": name}, thought="reading it"),
        )

    async def _observation(self, runner: Runner, agent: Agent, loaded: LoadedSkills | None) -> Any:
        events = [
            e
            async for e in runner._execute_tool(
                agent.tools[0],
                self._state(agent, loaded),
                self._context(),
                self._call_event("sql-style"),
            )
        ]
        ends = [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
        assert len(ends) == 1
        return ends[0].action.observation

    @pytest.mark.asyncio
    async def test_skill_view_reads_the_turn_load_instead_of_the_source(
        self, runner: Runner
    ) -> None:
        """The whole point: a body already in hand costs no second read."""
        source = CountingSource()
        agent = _agent("MainAgent", skills=SkillsRuntime(source=source))
        assert [t.name for t in agent.tools] == ["skill_view", "skill_search"]

        loaded = await source.load(SkillQuery())
        source.calls = 0

        observation = await self._observation(runner, agent, loaded)

        assert "use CTEs" in str(observation)
        assert source.calls == 0, "the tool went back to the source despite the turn's load"

    @pytest.mark.asyncio
    async def test_skill_search_searches_the_turn_load_too(self, runner: Runner) -> None:
        """The same keyword reaches ``skill_search``, so a search reads no file."""
        source = CountingSource()
        agent = _agent("MainAgent", skills=SkillsRuntime(source=source))
        loaded = await source.load(SkillQuery())
        source.calls = 0
        event = BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author="MainAgent",
            content=[],
            action=ToolAction(name="skill_search", parameters={"query": "sql"}, thought="finding"),
        )

        events = [
            e
            async for e in runner._execute_tool(
                agent.tools[1], self._state(agent, loaded), self._context(), event
            )
        ]

        ends = [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
        assert "- sql-style:" in str(ends[0].action.observation)
        assert source.calls == 0, "the search went back to the source despite the turn's load"

    @pytest.mark.asyncio
    async def test_without_a_turn_load_the_tool_still_works(self, runner: Runner) -> None:
        """An agent the runner never loaded for must not lose its skills entirely.

        ``state.skills`` is keyed by agent name and a tool can be reached on a path
        that filled no entry; ``.get`` then yields ``None``, which has to mean
        "nobody preloaded" and not "no such skill".
        """
        source = InlineFetchSource()
        agent = _agent("MainAgent", skills=SkillsRuntime(source=source))

        observation = await self._observation(runner, agent, loaded=None)

        assert "use CTEs" in str(observation)
        assert source.fetches == 1
