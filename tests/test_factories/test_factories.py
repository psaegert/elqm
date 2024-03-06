import os
import textwrap
from unittest.mock import MagicMock, patch

import pytest
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter, HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from pytest import raises

from elqm.factories.document_loaders import DocumentLoaderFactory
from elqm.factories.embeddings import EmbeddingFactory
from elqm.factories.models import ModelFactory
from elqm.factories.output_parsers import OutputParserFactory
from elqm.factories.postprocessing import PostprocessorFactory
from elqm.factories.preprocessing import HTMLSplitter, PreprocessorFactory
from elqm.factories.prompts import PromptFactory
from elqm.factories.retrievers import RetrieverFactory
from elqm.factories.splitters import SplitterFactory
from elqm.models.passthrough_model import PassthroughModel
from elqm.postprocessing import AppendAllCitationPostprocessor, AppendAllPostprocessor
from elqm.preprocessing import DictSplitter, Preprocessor
from elqm.splitters import SemanticChunker, SparkNLPSplitter
from elqm.utils import cache_exists, clear_cache

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def test_directory_json() -> None:
    document_loader = DocumentLoaderFactory.get_document_loader(
        document_loader_name="directory_json",
        document_loader_args={
            "content_key": "text",
        },
        index_name="pytest"
    )

    assert type(document_loader) is DirectoryLoader

    clear_cache(index_name="pytest")
    assert not cache_exists('pytest', retriever=None)


def test_json() -> None:
    document_loader = DocumentLoaderFactory.get_document_loader(
        document_loader_name="json",
        document_loader_args={
            "content_key": "text",
        },
        index_name="pytest"
    )

    assert type(document_loader) is JSONLoader

    clear_cache(index_name="pytest")
    assert not cache_exists('pytest', retriever=None)


def test_gpt4all_embeddings() -> None:
    embeddings = EmbeddingFactory.get_embedding(
        embedding_name="gpt4all"
    )

    assert isinstance(embeddings, GPT4AllEmbeddings)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large models in GitHub Actions")
def test_bge_large_embeddings() -> None:
    embeddings = EmbeddingFactory.get_embedding(
        embedding_name="BAAI/bge-large-en-v1.5",
        embedding_kwargs={
            "cuda_enabled": False,
            "device": "cpu"
        }
    )

    assert isinstance(embeddings, HuggingFaceEmbeddings)


@patch('elqm.factories.models.is_model_installed', return_value=True)
def test_mistral_model(mock_is_model_installed: MagicMock) -> None:
    model, max_length = ModelFactory.get_model(
        model_name="mistral"
    )

    assert isinstance(model, Ollama)
    assert max_length == 4096


def test_passthrough_model() -> None:
    model, max_length = ModelFactory.get_model(
        model_name=""
    )

    assert isinstance(model, PassthroughModel)
    assert max_length == 1e10


def test_str_output_parser() -> None:
    output_parser = OutputParserFactory.get_output_parser(
        output_parser="str_output_parser"
    )

    assert isinstance(output_parser, StrOutputParser)


def test_append_all_postprocessor() -> None:
    postprocessor = PostprocessorFactory.get_postprocessor(
        postprocessing_config_name="append_all_postprocessor"
    )

    assert isinstance(postprocessor, AppendAllPostprocessor)


def test_append_all_citation_postprocessor() -> None:
    postprocessor = PostprocessorFactory.get_postprocessor(
        postprocessing_config_name="append_all_citation_postprocessor"
    )

    assert isinstance(postprocessor, AppendAllCitationPostprocessor)


def test_preprocessing_dict_splitter() -> None:
    preprocessor = PreprocessorFactory.get_preprocessor(
        preprocessor_name="dict_splitter",
        splitter="recursive_character_splitter",
        splitter_kwargs={
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separators": [" "]
        },
        split_field="text"
    )

    assert type(preprocessor) is DictSplitter


def test_preprocessing_factory_remove_html() -> None:
    preprocessor = PreprocessorFactory.get_preprocessor(
        preprocessor_name="remove_html_tags_preprocessor"
    )

    assert isinstance(preprocessor, Preprocessor)


def test_preprocessing_factory_footnotes() -> None:
    preprocessor = PreprocessorFactory.get_preprocessor(
        preprocessor_name="transform_footnotes_preprocessor"
    )

    assert isinstance(preprocessor, Preprocessor)


def test_preprocessing_factory_raiseError() -> None:
    with raises(NotImplementedError):
        PreprocessorFactory.get_preprocessor(preprocessor_name="XXX")


def test_prompt_factory() -> None:
    prompt = PromptFactory.get_prompt(
        prompt_name="simple_history_v1"
    )

    assert isinstance(prompt, ChatPromptTemplate)


def test_retriever_factory() -> None:
    document_list = [
        Document(page_content="This is a test document",
                 metadata={"title": "Test Document 1"}),
        Document(page_content="The quick brown fox jumps over the lazy dog", metadata={
                 "title": "Test Document 2"}),
        Document(page_content="May the coffee kick in before reality does",
                 metadata={"title": "Test Document 3"}),
        Document(page_content="I am not lazy, I am on energy saving mode",
                 metadata={"title": "Test Document 4"}),
        Document(page_content="Time to get creative",
                 metadata={"title": "Test Document 5"}),
        Document(page_content="https://www.twitch.tv/wirtual",
                 metadata={"title": "Test Document 6"}),
        Document(page_content="Maybe this is enough documents",
                 metadata={"title": "Test Document 7"}),
    ]

    retriever = RetrieverFactory.get_retriever(
        retriever_name="FAISS",
        settings_file_for_dynaconf="pytest.test",
        index_name="pytest",
        cache=False,
        embeddings_object=EmbeddingFactory.get_embedding(
            embedding_name="gpt4all"),
        documents=document_list,
        retriever_args={"k_retrieved_documents": 5}
    )

    assert isinstance(retriever, VectorStoreRetriever)

    results = retriever.invoke("coffee")

    assert len(results) == 5
    assert results[0].metadata["title"] == "Test Document 3"

    # Clean up
    clear_cache(index_name="pytest")
    assert not cache_exists('pytest', retriever=None)


def test_retriever_BM25_retriever() -> None:
    document_list = [
        Document(page_content="This is a test document",
                 metadata={"title": "Test Document 1"}),
        Document(page_content="The quick brown fox jumps over the lazy dog", metadata={
                 "title": "Test Document 2"}),
        Document(page_content="May the coffee kick in before reality does",
                 metadata={"title": "Test Document 3"}),
        Document(page_content="I am not lazy, I am on energy saving mode",
                 metadata={"title": "Test Document 4"}),
        Document(page_content="Time to get creative",
                 metadata={"title": "Test Document 5"}),
        Document(page_content="https://www.twitch.tv/wirtual",
                 metadata={"title": "Test Document 6"}),
        Document(page_content="Maybe this is enough documents",
                 metadata={"title": "Test Document 7"}),
    ]

    bm25_retriever = RetrieverFactory.get_retriever(
        retriever_name="BM25",
        settings_file_for_dynaconf="pytest.test",
        index_name="pytest",
        cache=False,
        documents=document_list,
        retriever_args={
            "k_retrieved_documents": 2,
            "BM25_params": {
                "k1": 1.5,
                "b": 0.75,
                "epsilon": 0.25
            }
        },
    )

    assert isinstance(bm25_retriever, BaseRetriever)
    assert (bm25_retriever.invoke("Time to get creative")[0].page_content == "Time to get creative")

    clear_cache(index_name="pytest")
    assert not cache_exists('pytest', retriever=None)


def test_retriever_ensemble_retriever() -> None:
    document_list = [
        Document(page_content="This is a test document",
                 metadata={"title": "Test Document 1"}),
        Document(page_content="This is another test document", metadata={
                 "title": "Test Document 2"}),
        Document(page_content="May the coffee kick in before reality does",
                 metadata={"title": "Test Document 3"})
    ]

    ensemble_retriever = RetrieverFactory.get_retriever(
        retriever_name="ensemble",
        documents=document_list,
        retriever_args={
            "retriever_weights": [0.5, 0.5],
            "retrievers": [
                {
                    "retriever": "BM25",
                    "retriever_args": {
                        "k_retrieved_documents": 2,
                        "BM25_params": {
                            "k1": 1.5,
                            "b": 0.75,
                            "epsilon": 0.25
                        }
                    }
                },
                {
                    "retriever": "FAISS",
                    "retriever_args": {
                        "k_retrieved_documents": 2
                    }
                }
            ]},
        settings_file_for_dynaconf="pytest.test",
        index_name="pytest",
        cache=False,
        embeddings_object=EmbeddingFactory.get_embedding(
            embedding_name="gpt4all")
    )

    result = ensemble_retriever.invoke("test document")
    assert isinstance(ensemble_retriever, BaseRetriever)
    assert len(result) == 2
    assert (result[0].page_content == "This is a test document" or result[0].page_content == "This is another test document")
    assert (result[1].page_content == "This is another test document" or result[1].page_content == "This is a test document")

    clear_cache(index_name="pytest")
    assert not cache_exists('pytest', retriever=None)


def test_splitter_factory_recursive_splitter() -> None:
    splitter = SplitterFactory.get_splitter(
        splitter_name="recursive_character_splitter",
        separators=[" "],
        chunk_size=1,
        chunk_overlap=0,
    )

    assert isinstance(splitter, RecursiveCharacterTextSplitter)

    text = "This is a test document"
    chunks = splitter.split_text(text)

    words = ["This", " is", " a", " test", " document"]
    for i, chunk in enumerate(chunks):
        assert chunk in words


def test_splitter_factory_character_splitter() -> None:
    splitter = SplitterFactory.get_splitter(
        splitter_name="character_splitter",
        chunk_size=5,
        chunk_overlap=1,
        separator="",
    )

    assert isinstance(splitter, CharacterTextSplitter)

    text = "AAAAABBBBCCCCDDDD"
    chunks = splitter.split_text(text)

    words = ["AAAAA", "ABBBB", "BCCCC", "CDDDD"]
    for i, chunk in enumerate(chunks):
        assert chunk in words


def test_splitter_factory_token_splitter() -> None:
    splitter = SplitterFactory.get_splitter(
        splitter_name="token_splitter",
        chunk_size=1,
        chunk_overlap=0,
    )

    assert isinstance(splitter, TokenTextSplitter)

    text = "This is a test document"
    chunks = splitter.split_text(text)

    words = ["This", " is", " a", " test", " document"]
    for i, chunk in enumerate(chunks):
        assert chunk in words


def test_splitter_factory_semantic_splitter() -> None:
    splitter = SplitterFactory.get_splitter(
        splitter_name="semantic_splitter",
        embeddings="gpt4all",
        embedding_args={},
        percentile=40
    )

    assert isinstance(splitter, SemanticChunker)

    text = "The shimmering moonlight danced across the tranquil lake, casting a spell of serenity upon the night. A gentle breeze whispered through the trees, carrying with it the scent of pine and dew-kissed grass. In the distance, the faint sound of crickets filled the air, their rhythmic chirping adding to the symphony of the night. As I stood there, bathed in the soft glow of the moon, I couldn't help but feel a sense of peace wash over me, as if all the worries of the world had melted away in the beauty of this moment."

    chunks = splitter.split_text(text)

    assert len(chunks) >= 1
    assert type(chunks) is list
    assert type(chunks[0]) is str


def test_splitter_factory_sparkNLP() -> None:
    splitter = SplitterFactory.get_splitter(
        splitter_name="sparkNLP", chunk_size=15, chunk_overlap=7, separators=[" "])

    assert isinstance(splitter, SparkNLPSplitter)

    # text = 633 characters, 15 chunk_size with overlap --> so len(chunks) should be > 42
    text = "The shimmering moonlight danced across the tranquil lake, casting a spell of serenity upon the night. A gentle breeze whispered through the trees, carrying with it the scent of pine and dew-kissed grass. In the distance, the faint sound of crickets filled the air, their rhythmic chirping adding to the symphony of the night. As I stood there, bathed in the soft glow of the moon, I couldn't help but feel a sense of peace wash over me, as if all the worries of the world had melted away in the beauty of this moment."

    chunks = splitter.split_text(text)

    assert len(chunks) >= 42
    assert type(chunks) is list
    assert type(chunks[0]) is str


def test_splitter_factory_html_splitter() -> None:
    splitter = PreprocessorFactory.get_preprocessor(
        preprocessor_name="html_splitter",
        headers_to_split_on={
            "h1": "h1",
            "h2": "h3"
        },
        split_field="html"
    )

    assert isinstance(splitter, HTMLSplitter)
    assert isinstance(splitter.splitter, HTMLHeaderTextSplitter)

    raw_dict = {
        "example_celex_id":
            {"html": textwrap.dedent("""<!DOCTYPE html>
                <html>
                <body>
                    <div>
                        <h1>Foo</h1>
                        <p>Some intro text about Foo.</p>
                        <div>
                            <h2>Bar main section</h2>
                            <p>Some intro text about Bar.</p>
                            <h3>Bar subsection 1</h3>
                            <p>Some text about the first subtopic of Bar.</p>
                            <h3>Bar subsection 2</h3>
                            <p>Some text about the second subtopic of Bar.</p>
                        </div>
                        <div>
                            <h2>Baz</h2>
                            <p>Some text about Baz</p>
                        </div>
                        <br>
                        <p>Some concluding text about Foo</p>
                    </div>
                </body>
                </html>
                """)}}

    preprocessed_dict = splitter.preprocess(raw_dict)

    assert len(preprocessed_dict) >= 1
    assert type(preprocessed_dict) is dict
    assert type(preprocessed_dict["example_celex_id_0"]) is dict
    assert preprocessed_dict["example_celex_id_0"]["html"] == "Foo"
    assert preprocessed_dict["example_celex_id_1"]["html"] == "Some intro text about Foo.  \nBar main section Bar subsection 1 Bar subsection 2"
    assert preprocessed_dict["example_celex_id_1"]["HTMLSplitter_h1"] == "Foo"
