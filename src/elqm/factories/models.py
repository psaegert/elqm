import warnings
from typing import Any

from langchain_community.llms.ollama import Ollama
from langchain_core.language_models.llms import BaseLLM

from elqm.models.passthrough_model import PassthroughModel
from elqm.utils import is_model_installed


class ModelFactory:
    @staticmethod
    def get_model(model_name: str, *args: Any, **kwargs: Any) -> tuple[BaseLLM, int]:
        """
        Factory method to get a model

        Parameters
        ----------
        model_name : str
            Name of the model to use. Determines which model class to instantiate.

        Returns
        -------
        BaseLLM
            An instance of the model class specified by model_name.
        int
            The maximum length of the model input.
        """
        if model_name != "" and not is_model_installed(model_name):
            raise ValueError(f"Model {model_name} is not installed. Install it with `ollama pull {model_name}` or check for models on https://ollama.ai/library?sort=newest")

        # FIXME: Maximum content length is not always 4096
        match model_name:
            case "llama2":
                return Ollama(model="llama2", *args, **kwargs), 4096
            case "mistral":
                return Ollama(model="mistral", *args, **kwargs), 4096
            case "phi":
                return Ollama(model="phi", *args, **kwargs), 4096
            case "orca2":
                return Ollama(model="orca2", *args, **kwargs), 4096
            case "mixtral":
                return Ollama(model="mixtral", *args, **kwargs), 4096
            case "gemma":
                return Ollama(model="gemma", *args, **kwargs), 4096
            case "":
                return PassthroughModel(*args, **kwargs), 1e10  # type: ignore
            case _:
                warnings.warn(f"\033[93mModel '{model_name}' not found. Will use default model instead.\033[0m")
                return Ollama(model="llama2", *args, **kwargs), 4096
