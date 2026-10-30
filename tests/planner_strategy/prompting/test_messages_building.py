from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents.llm import LLMMessage
from domyn.agents.planner_strategy.prompting.messages_building import MessageBuilder


class DummyMessageBuilder(MessageBuilder):
    def get_response_message(self, event: BaseEvent, use_stop: bool) -> LLMMessage:
        return LLMMessage(role=self.assistant_role, content="")

    def get_agent_call_messages(self, event: BaseEvent) -> list[LLMMessage]:
        return []

    def get_agent_start_messages(self, event: BaseEvent) -> list[LLMMessage]:
        return []

    def get_agent_end_messages(self, event: BaseEvent) -> list[LLMMessage]:
        return []

    def get_tool_end_messages(self, event: BaseEvent) -> list[LLMMessage]:
        return []

    def get_tool_error_messages(self, event: BaseEvent) -> list[LLMMessage]:
        return []

    def add_start_message(self, messages: list[LLMMessage], agent) -> list[LLMMessage]:
        return messages


def test_get_user_request_message_with_user_info_dict() -> None:
    builder = DummyMessageBuilder()
    event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Can you check my order?")],
        metadata={"user_info": {"tier": "gold", "region": "eu"}},
    )

    message = builder.get_user_request_message(event=event)

    assert message.role == "user"
    assert (
        message.content
        == "<INFO>\nTIER: gold\nREGION: eu\n</INFO>\n\nUSER INPUT: Can you check my order?"
    )


def test_get_user_request_message_with_non_dict_user_info() -> None:
    builder = DummyMessageBuilder()
    event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Need help")],
        metadata={"user_info": 12345},
    )

    message = builder.get_user_request_message(event=event)

    assert message.content == "<INFO>\n12345</INFO>\n\nUSER INPUT: Need help"


def test_get_user_request_message_without_user_info() -> None:
    builder = DummyMessageBuilder()
    event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Hello")],
        metadata={},
    )

    message = builder.get_user_request_message(event=event)

    assert message.content == "USER INPUT: Hello"


def test_get_user_request_message_with_custom_user_info_key() -> None:
    builder = DummyMessageBuilder(user_info_key="profile")
    event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Status update")],
        metadata={"profile": {"name": "Alice"}},
    )

    message = builder.get_user_request_message(event=event)

    assert message.content == "<INFO>\nNAME: Alice\n</INFO>\n\nUSER INPUT: Status update"


def test_get_user_request_message_with_falsy_user_info_value() -> None:
    builder = DummyMessageBuilder()
    event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Hello again")],
        metadata={"user_info": ""},
    )

    message = builder.get_user_request_message(event=event)

    assert message.content == "USER INPUT: Hello again"


def test_get_user_request_message_with_none_user_input_text() -> None:
    builder = DummyMessageBuilder()
    event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text=None)],
        metadata={"user_info": {"channel": "chat"}},
    )

    message = builder.get_user_request_message(event=event)

    assert message.content == "<INFO>\nCHANNEL: chat\n</INFO>\n\nUSER INPUT: None"


def test_get_user_request_message_with_custom_patterns_and_role() -> None:
    builder = DummyMessageBuilder(
        info_start_pattern="<META>", info_end_pattern="</META>", user_role="customer"
    )
    event = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="Need invoice")],
        metadata={"user_info": {"company": "Acme"}},
    )

    message = builder.get_user_request_message(event=event)

    assert message.role == "customer"
    assert message.content == "<META>\nCOMPANY: Acme\n</META>\n\nUSER INPUT: Need invoice"
