import copy
import re
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import calculate_cosine_distances, combine_sentences


class SemanticChunker(BaseDocumentTransformer):
    """Splits the text based on semantic similarity.

    Taken from Greg Kamradt's wonderful notebook:
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/5_Levels_Of_Text_Splitting.ipynb

    All credit to him.

    At a high level, this splits into sentences, then groups into groups of 3
    sentences, and then merges one that are similar in the embedding space.
    """

    def __init__(self, embeddings: Embeddings, add_start_index: bool = False, percentile: float = 95):
        self._add_start_index = add_start_index
        self.embeddings = embeddings
        self.breakpoint_percentile_threshold = percentile

    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""
        # Splitting the essay on '.', '?', and '!'
        single_sentences_list = re.split(r"(?<=[.?!])\s+", text)
        sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]
        sentences = combine_sentences(sentences)
        embeddings = self.embeddings.embed_documents(
            [x["combined_sentence"] for x in sentences]
        )
        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i]
        distances, sentences = calculate_cosine_distances(sentences)
        start_index = 0

        if len(distances) == 0:
            return [text]

        # Create a list to hold the grouped sentences
        chunks = []
        breakpoint_distance_threshold = np.percentile(
            distances, self.breakpoint_percentile_threshold
        )  # If you want more chunks, lower the percentile cutoff

        indices_above_thresh = [
            i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
        ]  # The indices of those breakpoints on your list

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index: end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])  # type: ignore
            chunks.append(combined_text)

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])  # type: ignore
            chunks.append(combined_text)
        return chunks

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))
