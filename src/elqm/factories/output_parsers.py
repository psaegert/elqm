import warnings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.transform import BaseTransformOutputParser


class OutputParserFactory:
    @staticmethod
    def get_output_parser(output_parser: str) -> BaseTransformOutputParser[str]:
        """
        Factory method to get an output parser.

        Parameters
        ----------
        output_parser : str
            Name of the output parser to use. Determines which output parser class to instantiate.

        Returns
        -------
        BaseTransformOutputParser[str]
            An instance of the output parser class specified by output_parser.
        """
        match output_parser:
            case "str_output_parser":
                return StrOutputParser()
            case _:
                warnings.warn(f"\033[93mOutput parser '{output_parser}' not found. Will use default output parser instead.\033[0m")
                return StrOutputParser()
