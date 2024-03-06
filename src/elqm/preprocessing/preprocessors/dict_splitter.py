import os

from langchain.text_splitter import TextSplitter
from tqdm import tqdm

from elqm.preprocessing.preprocessors.preprocessor import Preprocessor


class DictSplitter(Preprocessor):
    """
    Wrapper for langchain TextSplitter to split the documents formatted as a dictionary.

    Parameters
    ----------
    splitter : TextSplitter
        The splitter to use to split the documents.
    split_field : str
        The field of the document to split.
    """
    def __init__(self, splitter: TextSplitter, split_field: str):
        self.splitter = splitter
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
        for filename, document in tqdm(document_dict.items(), desc=f"Splitting Documents with {self.splitter.__class__.__name__}", disable=not verbose):
            # Split the field of the document into chunks with the splitter
            chunks = self.splitter.split_text(document[self.split_field])

            # Create a new document for each chunk
            for i, chunk in enumerate(chunks):
                basename, extension = os.path.splitext(os.path.basename(filename))
                new_dict[f"{basename}_{i}{extension}"] = document.copy()
                new_dict[f"{basename}_{i}{extension}"][self.split_field] = chunk
                new_dict[f"{basename}_{i}{extension}"]["ID"] = uid

                uid += 1

        return new_dict
