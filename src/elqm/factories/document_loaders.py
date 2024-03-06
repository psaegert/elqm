import os
import warnings
from typing import Any

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.json_loader import JSONLoader

from elqm.utils import extract_metadata, get_dir


class DocumentLoaderFactory:
    @staticmethod
    def get_document_loader(document_loader_name: str, *args: Any, **kwargs: Any) -> BaseLoader:
        """
        Factory method to get a document loader

        Parameters
        ----------
        document_loader_name : str
            Name of the document loader to use. Determines which loader class to instantiate.
        *args : Any
            Positional arguments to pass to the loader class.
        **kwargs : Any
            Keyword arguments to pass to the loader class.

        Returns
        -------
        BaseLoader
            An instance of the loader class specified by document_loader_name.
        """
        match document_loader_name:
            case "json":
                preprocessing_cache_dir = get_dir("cache", kwargs["index_name"], "preprocessed_documents", create=True)
                return JSONLoader(
                    file_path=os.path.join(preprocessing_cache_dir, "preprocessed_data.json"),
                    jq_schema=".[]",
                    content_key=kwargs["document_loader_args"]["content_key"],
                    metadata_func=lambda r, m: extract_metadata(r, m, mapping=kwargs.get("mapping", None)))
            case "directory":
                return DirectoryLoader(*args, **kwargs)
            case "directory_json":
                preprocessing_cache_dir = get_dir("cache", kwargs["index_name"], "preprocessed_documents", create=True)
                return DirectoryLoader(
                    preprocessing_cache_dir,
                    glob='**/*.json',
                    show_progress=True,
                    loader_cls=JSONLoader,  # type: ignore
                    loader_kwargs={
                        "jq_schema": ".",
                        "content_key": kwargs["document_loader_args"]["content_key"],
                        "metadata_func": lambda r, m: extract_metadata(r, m, mapping=kwargs.get("mapping", None))})
            case _:
                warnings.warn(f"\033[93m Document loader '{document_loader_name}' not found. Will use default document loader instead.\033[0m")
                return DirectoryLoader(*args, **kwargs)
