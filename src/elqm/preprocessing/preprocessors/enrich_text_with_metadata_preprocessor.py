from typing import Any

from tqdm import tqdm

from elqm.preprocessing.preprocessors.preprocessor import Preprocessor


class EnrichTextWithMetadataPreprocessor(Preprocessor):
    """
    For enriching the text of the documents with metadata.
    """
    def __init__(self, metadata_keys: list, load_key: str, save_key: str) -> None:
        """
        Args:
            metadata: List of metadata keys to enrich the text with
            load_key: The key of the document to load the text from
            save_key: The key of the document to save the enriched text to
        """
        self.metadata_keys = metadata_keys
        self.load_key = load_key
        self.save_key = save_key

    def __add_metadata_to_text(self, metadata: Any, metadata_key: str) -> str:
        """
        Adds metadata to the text of the documents at the end of the text.

        Args:
            text: The text to enrich with metadata.
            metadata: The metadata to add to the text.
            metadata_key: The key of the metadata.

        Returns:
            str: The enriched text.
        """
        if (type(metadata) is list):
            metadata_str = [str(item) for item in metadata]
            return f"\n{metadata_key}:" + ",".join(metadata_str)
        else:
            return f"\n{metadata_key}:{metadata}"

    def preprocess(self, document_dict: dict[str, dict], verbose: bool = True) -> dict[str, dict]:
        """
        Adds metadata to the text of the documents at the end of the text.

        Args:
            document_dict: The list of texts to preprocess.
            verbose: The verbosity of the preprocessing.

        Returns:
            dict[str, dict]The list of preprocessed texts.
        """
        preprocessed_document_dict = document_dict.copy()

        for id, document in tqdm(preprocessed_document_dict.items(), disable=not verbose, desc="Enriching Text with Metadata"):
            preprocessed_document_dict[id][self.save_key] = f"Document content:\n{document[self.load_key]}"
            for metadata_key in self.metadata_keys:
                keys = metadata_key.split('.')
                value = document
                final_key = ""
                for key in keys:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                        final_key = key
                    else:
                        print(f"Warning: Metadata key '{metadata_key}' not found in document '{id}'. Will be skipped.") if verbose else None
                        break
                else:
                    preprocessed_document_dict[id][self.save_key] += self.__add_metadata_to_text(value, final_key)

        return preprocessed_document_dict
