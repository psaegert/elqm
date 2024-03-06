import warnings
from typing import Any

from elqm.postprocessing import AppendAllCitationPostprocessor, AppendAllPostprocessor, Postprocessor


class PostprocessorFactory:
    @staticmethod
    def get_postprocessor(postprocessing_config_name: str, *args: Any, **kwargs: Any) -> Postprocessor:
        """
        Factory method to get a postprocessor

        Parameters
        ----------
        postprocessing_config_name : str
            Name of the postprocessor to use. Determines which postprocessor class to instantiate.
        *args : Any
            Positional arguments to pass to the postprocessor class.
        **kwargs : Any
            Keyword arguments to pass to the postprocessor class.

        Returns
        -------
        Postprocessor
            An instance of the postprocessor class specified by postprocessing_config_name.
        """
        match postprocessing_config_name:
            case "append_all_postprocessor":
                return AppendAllPostprocessor(*args, **kwargs)
            case "append_all_citation_postprocessor":
                return AppendAllCitationPostprocessor(*args, **kwargs)
            case _:
                warnings.warn(f"\033[93m '{postprocessing_config_name}' not found. Will use default postprocessor instead.\033[0m")
                return AppendAllPostprocessor(*args, **kwargs)
