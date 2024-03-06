from typing import Any

from langchain.prompts.chat import ChatPromptValue
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult
from langchain_core.outputs.generation import Generation


class PassthroughModel(BaseLLM):
    """
    A passthrough model that returns the input as the output.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def _llm_type(self) -> str:  # abstract method, needs to be implemented
        return "pass_through"

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:  # abstract method, needs to be implemented
        return LLMResult(generations=[[Generation(text=prompt) for prompt in prompts]])

    def invoke(self, input: ChatPromptValue) -> str | list[str | dict[Any, Any]]:  # type: ignore
        return input.messages[-1].content  # The last (human) message with the retrieved documents.
