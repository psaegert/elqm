import os

from langchain.text_splitter import HTMLHeaderTextSplitter

from elqm.preprocessing.preprocessors.preprocessor import Preprocessor


class HTMLSplitter(Preprocessor):
    """
    Wrapper for langchain HTMLHeaderTextSplitter to split the documents formatted as a dictionary.

    Parameters
    ----------
    headers_to_split_on : list[tuple[str, str]]
        The headers to split on.
    split_field : str
        The field of the document to split.
    """
    def __init__(self, headers_to_split_on: list[tuple[str, str]], split_field: str) -> None:
        self.splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        self.split_field = split_field

    def preprocess(self, document_dict: dict[str, dict], verbose: bool = False) -> dict[str, dict]:
        """
        Splits the documents in the dict with (filename: document) into a larger dictionary with (filename: smaller_document)

        Parameters
        ----------
        document_dict : dict[str, dict]
            The dictionary containing the documents to split with (filename: document)
        verbose : bool
            The verbosity of the preprocessing.

        Returns
        -------
        dict[str, dict]
            The dictionary containing the split documents with (filename: smaller_document)
        """

        new_dict: dict[str, dict] = {}
        uid = 0
        for filename, document in document_dict.items():
            # Split the field of the document into chunks with the splitter
            # This is a list of Documents!
            chunks = self.splitter.split_text(document[self.split_field])

            # Create a new document for each chunk
            for i, chunk in enumerate(chunks):
                basename, extension = os.path.splitext(os.path.basename(filename))
                new_dict[f"{basename}_{i}{extension}"] = document.copy()
                new_dict[f"{basename}_{i}{extension}"][self.split_field] = chunk.page_content

                for k, v in chunk.metadata.items():
                    new_dict[f"{basename}_{i}{extension}"][f"{self.__class__.__name__}_{k}"] = v

                new_dict[f"{basename}_{i}{extension}"]["ID"] = uid

                uid += 1

        return new_dict
