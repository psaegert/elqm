import os
import pickle
from typing import Any

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from tqdm import tqdm

from elqm.factories.models import ModelFactory
from elqm.utils import get_cache_filename, get_dir


class RetrieverFactory:
    """
    Factory class for retrievers
    """
    @staticmethod
    def get_retriever(retriever_name: str, *args: Any, **kwargs: Any) -> BaseRetriever:
        """
        Factory method to get a retriever

        Parameters
        ----------
        retriever_name : str
            Name of the retriever to use. Determines which retriever class to instantiate.
        *args : Any
            Positional arguments to pass to the retriever class.
        **kwargs : Any
            Keyword arguments to pass to the retriever class.

        Returns
        -------
        VectorStoreRetriever
            An instance of the retriever class specified by retriever_name.
        """
        match retriever_name:
            case "FAISS":
                return RetrieverFactory.build_faiss_retriever(*args, **kwargs)
            case "self_query_chroma":
                raise NotImplementedError("We have not found an LLM for stable self querying yet.")
                return RetrieverFactory.build_self_query_chroma_retriever(*args, **kwargs)
            case "BM25":
                return RetrieverFactory.build_bm25_retriever(*args, **kwargs)
            case "ensemble":
                return RetrieverFactory.build_ensemble_retriever(*args, **kwargs)
            case _:
                raise ValueError(f"Retriever {retriever_name} not supported")

    @staticmethod
    def build_faiss_retriever(*args: Any, **kwargs: Any) -> VectorStoreRetriever:
        index_dir = os.path.join(get_dir("cache", kwargs["index_name"], "FAISS", create=True))
        index_file = os.path.join(index_dir, get_cache_filename("FAISS"))

        if kwargs["cache"] and os.path.exists(index_file):
            print("Loading FAISS index from cache")
            vector_store = FAISS.load_local(index_dir, kwargs["embeddings_object"])
        else:
            print("Creating FAISS index")
            batch_size = min(100, len(kwargs["documents"]))
            batches = [kwargs["documents"][i:i + batch_size] for i in range(0, len(kwargs["documents"]), batch_size)]

            vector_stores = [FAISS.from_documents(batch, kwargs["embeddings_object"]) for batch in tqdm(batches, desc="Creating FAISS vectorstores")]
            vector_store = vector_stores[0]

            for vec in vector_stores[1:]:
                vector_store.merge_from(vec)

            vector_store.save_local(index_dir, "index")

        retriever = vector_store.as_retriever(search_kwargs={'k': kwargs["retriever_args"]["k_retrieved_documents"]})
        return retriever

    @staticmethod
    def build_self_query_chroma_retriever(*args: Any, **kwargs: Any) -> SelfQueryRetriever:
        index_dir = os.path.join(get_dir("cache", kwargs["index_name"], "chroma", create=True))
        index_file = os.path.join(index_dir, get_cache_filename("chroma"))

        if "documents" in kwargs and kwargs["documents"] is not None:
            for document in kwargs["documents"]:
                if document.page_content is None:
                    document.page_content = ""  # HACK: This is temporary. Investigate and fix in document loaders

        if kwargs["cache"] and os.path.exists(index_file):
            print("Loading Chroma index from cache")
            vectorstore = Chroma(persist_directory=index_dir, embedding_function=kwargs["embeddings_object"])
        else:
            print("Creating Chroma index")
            vectorstore = Chroma.from_documents(
                documents=kwargs["documents"],
                persist_directory=index_dir,
                embedding=kwargs["embeddings_object"])

        metadata_field_info = [AttributeInfo(name=key, type=value["type"], description=value["description"]) for key, value in kwargs["mapping"].items()]

        return SelfQueryRetriever.from_llm(
            llm=ModelFactory.get_model(kwargs["model"], temperature=0)[0],  # HACK: This may cause circular imports in the future
            vectorstore=vectorstore,
            document_contents=kwargs["document_content_description"],
            metadata_field_info=metadata_field_info,
            enable_limit=True,
            structured_query_translator=ChromaTranslator())

    @staticmethod
    def build_bm25_retriever(*args: Any, **kwargs: Any) -> BM25Retriever:
        bm25_dir = os.path.join(get_dir("cache", kwargs["index_name"], "BM25", create=True))
        bm25_file = os.path.join(bm25_dir, get_cache_filename("BM25"))

        if kwargs["cache"] and os.path.exists(bm25_file):
            print("Loading BM25 retriever from cache")
            with open(bm25_file, "rb") as f:
                bm25_retriever = pickle.load(f)
        else:
            print("Creating BM25 retriever")
            bm25_retriever = BM25Retriever.from_documents(
                documents=kwargs["documents"],
                bm25_params=kwargs["retriever_args"]["BM25_params"],
            )
            bm25_retriever.k = kwargs["retriever_args"]["k_retrieved_documents"]

            with open(bm25_file, "wb") as f:
                pickle.dump(bm25_retriever, f)

        return bm25_retriever

    @staticmethod
    def build_ensemble_retriever(*args: Any, **kwargs: Any) -> EnsembleRetriever:
        retrievers = []

        for retriever in kwargs['retriever_args']['retrievers']:
            # Replace the retrievers of the config (kwargs parameter) with the retriever of the ensemble
            # Thus, the factory methods can read the rest of the config as if it was the only retriever
            kwargs_copy = kwargs.copy()
            kwargs_copy['retriever'] = retriever['retriever']
            kwargs_copy['retriever_args'] = retriever['retriever_args']

            retrievers.append(RetrieverFactory.get_retriever(kwargs_copy['retriever'], **kwargs_copy))

        ensemble_retriever = EnsembleRetriever(
            retrievers=retrievers,  # type: ignore
            weights=kwargs["retriever_args"]["retriever_weights"]
        )

        return ensemble_retriever
