import json
import os

from tqdm import tqdm

from elqm.factories.preprocessing import PreprocessorFactory
from elqm.preprocessing.preprocessors import Preprocessor
from elqm.utils import get_dir


class PreprocessingPipeline:
    """
    Pipeline to preprocess documents before loading and splitting

    Attributes
    ----------
    steps : list[Preprocessor]
        The preprocessing steps to run.
    debug : bool
        Whether to run in debug mode.
    """

    def __init__(self, steps: dict[str, dict], debug: bool = False):
        self.debug = debug
        self.steps: list[Preprocessor] = [PreprocessorFactory.get_preprocessor(
            preprocessor_name, **preprocessor_kwargs) for preprocessor_name, preprocessor_kwargs in steps.items()]

    def load_documents(self, document_dir: str, verbose: bool = True) -> dict[str, dict]:
        """
        Load documents from a directory

        Parameters
        ----------
        document_dir : str
            The directory to load the documents from.
        verbose : bool
            Whether to show a progress bar.

        Returns
        -------
        dict[str, dict]
            The loaded documents.
        """
        document_dict = {}

        for file in tqdm(os.listdir(document_dir), disable=not verbose, desc="Loading documents"):
            with open(os.path.join(document_dir, file), "r") as f:
                document_dict[file] = json.load(f)
                document_dict[file]["CELEX_ID"] = os.path.splitext(file)[0]  # HACK: This should ideally be done in the scraping already
                document_dict[file]["ID"] = document_dict[file]["CELEX_ID"]  # Used to uniquely identify the document

        return document_dict

    def save_documents(self, document_dict: dict[str, dict], document_dir: str, drop_keys: list[str], verbose: bool = True) -> None:
        """
        Save documents to a directory

        Parameters
        ----------
        document_dict : dict[str, dict]
            The documents to save.
        document_dir : str
            The directory to save the documents to.
        drop_keys : list[str]
            The keys to drop from the documents.
        verbose : bool
            Whether to show a progress bar.
        """
        for file_name, document in tqdm(document_dict.items(), disable=not verbose, desc="Saving documents"):
            for key in drop_keys:
                document.pop(key, None)
            with open(os.path.join(document_dir, file_name), "w") as f:
                json.dump(document, f)

    def preprocess(self, document_dict: dict[str, dict], verbose: bool = True) -> dict[str, dict]:
        """
        Preprocess documents with the preprocessing steps

        Parameters
        ----------
        document_dict : dict[str, dict]
            The documents to preprocess.
        verbose : bool
            Whether to show a progress bar.

        Returns
        -------
        dict[str, dict]
            The preprocessed documents.
        """
        for step in self.steps:
            # Input: dict (filename: document_dict)
            # Output: dict (filename: document_dict)
            document_dict = step.preprocess(document_dict, verbose=verbose)

            if self.debug:
                for filename, document in tqdm(document_dict.items(), disable=not verbose, desc=f"Saving files processed by {step.__class__.__name__}"):
                    step_dir = get_dir("debug", 'preprocessing', step.__class__.__name__, create=True)

                    with open(os.path.join(step_dir, filename), "w") as f:
                        json.dump(document, f)

        return document_dict
