import warnings
from typing import Any

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.embeddings import Embeddings


class EmbeddingFactory:
    @staticmethod
    def get_embedding(embedding_name: str, embedding_kwargs: Any | None = None) -> Embeddings:
        """
        Factory method to get an embedding

        Parameters
        ----------
        embedding_name : str
            Name of the document embedding to use. Determines which embedding class to instantiate.
        embedding_kwargs : Any, optional
            Keyword arguments for the embedding class, by default None

        Returns
        -------
        Embeddings
            An instance of the embedding class specified by embedding_name.
        """
        match embedding_name:
            case "gpt4all":
                return GPT4AllEmbeddings()  # type: ignore
            case "BAAI/bge-large-en-v1.5":
                if embedding_kwargs is None:
                    raise ValueError("embedding_kwargs must be provided for HuggingFaceEmbeddings")
                return HuggingFaceEmbeddings(
                    model_name="BAAI/bge-large-en-v1.5",
                    model_kwargs={'device': embedding_kwargs['device'] if embedding_kwargs['cuda_enabled'] else 'cpu'})
            case _:
                warnings.warn(f"\033[93mEmbedding '{embedding_name}' not found. Will use default embedding instead.\033[0m")
                return GPT4AllEmbeddings()  # type: ignore
