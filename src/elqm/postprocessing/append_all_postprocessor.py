from langchain_core.documents import Document

from .postprocessor import Postprocessor


class AppendAllPostprocessor(Postprocessor):
    """
    Postprocessor that appends all documents to a single string.

    Attributes
    ----------
    delimiter : str
        The delimiter to use between documents.
    """

    def __init__(self, delimiter: str = "\n") -> None:
        """
        Initializes the AppendAllPostprocessor.

        Parameters
        ----------
        delimiter : str
            The delimiter to use between documents.
        """
        self.delimiter = delimiter

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
        return f"{self.delimiter}".join([d.page_content for d in documents])
