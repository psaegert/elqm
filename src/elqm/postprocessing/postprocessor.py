from abc import abstractmethod

from langchain_core.documents import Document


class Postprocessor:
    """
    Abstract class for postprocessing of retrieved documents
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def postprocess(self, documents: list[Document]) -> str:
        """
        Postprocesses the document list retrieved by the retriever.

        Parameters
        ----------
        documents : list[Document]
            The list of documents to postprocess.

        Returns
        -------
        str
            A string constructed from the postprocessed documents for the LLM.
        """
        raise NotImplementedError
