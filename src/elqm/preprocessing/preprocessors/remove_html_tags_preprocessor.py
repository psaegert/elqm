import re

from bs4 import BeautifulSoup
from tqdm import tqdm

from elqm.preprocessing.preprocessors.preprocessor import Preprocessor


class RemoveHTMLTagsPreprocessor(Preprocessor):
    """
    Abstract class for preprocessing.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def preprocess(document_dict: dict[str, dict], verbose: bool = True) -> dict[str, dict]:
        """
        Removes the html markup from the text.

        Parameters
        ----------
        document_dict : dict[str, dict]
            The list of texts to preprocess.
        verbose : bool
            The verbosity of the preprocessing.

        Returns
        -------
        dict[str, dict]
            The list of preprocessed texts.
        """

        preprocessed_document_dict = document_dict.copy()
        for id, document in tqdm(preprocessed_document_dict.items(), disable=not verbose, desc="Removing HTML tags"):
            soup = BeautifulSoup(document['html'], 'html.parser')

            clean_text = soup.get_text(separator='\n')
            clean_text = re.sub(r'(\n\s*\n)+', '\n', clean_text)

            preprocessed_document_dict[id]['text'] = clean_text

        return preprocessed_document_dict
