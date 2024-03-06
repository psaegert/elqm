from abc import abstractmethod


class Preprocessor:
    """
    Abstract class for preprocessing.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def preprocess(document_dict: dict[str, dict], verbose: bool = False) -> dict[str, dict]:
        """
        Preprocesses the documents in the dict with (filename: document) and returns a dictionary with modified keys.

        Parameters
        ----------
        document_dict : dict[str, dict]
            The data dictionary to preprocess
        verbose : bool
            The verbosity of the preprocessing.

        Returns
        -------
        dict[str, dict]
            The preprocessed data dictionary.
        """
        raise NotImplementedError
