from abc import ABC, abstractmethod

from pydantic import BaseModel

from domyn.agents.core import BaseAgent, BaseEvent
from domyn.agents.llm import LLMMessage


class MessageBuilder(ABC, BaseModel):
    info_start_pattern: str = "<INFO>"
    info_end_pattern: str = "</INFO>"
    user_role: str = "user"
    assistant_role: str = "assistant"
    feedback_role: str = "user"
    user_info_key: str = "user_info"

    @abstractmethod
    def get_response_message(self, event: BaseEvent, use_stop: bool) -> LLMMessage: ...

    @abstractmethod
    def get_agent_call_messages(self, event: BaseEvent) -> list[LLMMessage]:
        """Messages representing a calling agent invoking a sub-agent (AGENT_CALL perspective)."""
        ...

    @abstractmethod
    def get_agent_start_messages(self, event: BaseEvent) -> list[LLMMessage]:
        """Messages representing a called agent receiving its task (AGENT_START perspective)."""
        ...

    @abstractmethod
    def get_agent_end_messages(self, event: BaseEvent) -> list[LLMMessage]: ...

    @abstractmethod
    def get_tool_end_messages(
        self, event: BaseEvent, forward_verbatim_enabled: bool = True
    ) -> list[LLMMessage]: ...

    @abstractmethod
    def get_tool_error_messages(self, event: BaseEvent) -> list[LLMMessage]: ...

    @abstractmethod
    def add_start_message(
        self, messages: list[LLMMessage], agent: BaseAgent
    ) -> list[LLMMessage]: ...

    def get_user_request_message(self, event: BaseEvent) -> LLMMessage:
        # TODO: only the first part's text is used; files, images and further parts are dropped.
        info = ""
        if user_info := event.metadata.get(self.user_info_key, {}):
            if isinstance(user_info, dict):
                for k, v in user_info.items():
                    info += f"{k.upper()}: {v}\n"
            else:
                info = str(user_info)

        message = f"{self.info_start_pattern}\n{info}{self.info_end_pattern}\n\n" if info else ""
        message += f"USER INPUT: {event.content[0].text}"

        return LLMMessage(role=self.user_role, content=message)
