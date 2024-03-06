import warnings
from typing import Any

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter, TokenTextSplitter

# Avoid circular import by specifying the full path
from elqm.factories.embeddings import EmbeddingFactory
from elqm.splitters import SemanticChunker, SparkNLPSplitter


class SplitterFactory:
    @staticmethod
    def get_splitter(splitter_name: str, *args: Any, **kwargs: Any) -> TextSplitter:
        """
        Factory method to get a splitter

        Parameters
        ----------
        splitter_name : str
            Name of the splitter to use. Determines which splitter class to instantiate.
        *args : Any
            Positional arguments to pass to the splitter class.
        **kwargs : Any
            Keyword arguments to pass to the splitter class.

        Returns
        -------
        TextSplitter
            An instance of the splitter class specified by splitter_name.
        """
        match splitter_name:
            case "recursive_character_splitter":
                return RecursiveCharacterTextSplitter(
                    separators=kwargs["separators"],
                    chunk_size=kwargs['chunk_size'],
                    chunk_overlap=kwargs['chunk_overlap'],
                    length_function=len,
                    is_separator_regex=False)
            case "sparkNLP":
                # Convert BoxList to a list of strings
                separator_list = [str(separator) for separator in kwargs["separators"]]
                return SparkNLPSplitter(chunk_size=kwargs['chunk_size'], chunk_overlap=kwargs['chunk_overlap'], separators=separator_list)
            case "character_splitter":
                return CharacterTextSplitter(
                    separator=kwargs["separator"],
                    chunk_size=kwargs['chunk_size'],
                    chunk_overlap=kwargs['chunk_overlap'],
                    length_function=len,
                    is_separator_regex=False)
            case "token_splitter":
                return TokenTextSplitter(
                    chunk_size=kwargs['chunk_size'],
                    chunk_overlap=kwargs['chunk_overlap'])
            case "semantic_splitter":
                # SemanticChunker.split_text still returns a list of strings
                return SemanticChunker(
                    embeddings=EmbeddingFactory.get_embedding(
                        kwargs["embeddings"],
                        embedding_kwargs=kwargs["embedding_args"]),
                    percentile=kwargs["percentile"])  # type: ignore
            case _:
                warnings.warn(
                    f"\033[93mSplitter '{splitter_name}' not found. Will use default splitter instead.\033[0m")
                return RecursiveCharacterTextSplitter(
                    separators=[" "],
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    is_separator_regex=False)
