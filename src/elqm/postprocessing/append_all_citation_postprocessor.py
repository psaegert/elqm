from langchain_core.documents import Document

from .postprocessor import Postprocessor


class AppendAllCitationPostprocessor(Postprocessor):
    """
    Postprocessor that appends all documents to a single string.

    Attributes
    ----------
    delimiter : str
        The delimiter to use between documents.
    """

    def __init__(self, delimiter: str = "\n", block_quotes: str = "```") -> None:
        """
        Initializes the AppendAllPostprocessor.

        Parameters
        ----------
        delimiter : str
            The delimiter to use between documents.
        block_quotes : str
            The block quotes to use for the documents.
        """
        self.delimiter = delimiter
        self.block_quotes = block_quotes

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
        document_strings = []

        for document in documents:
            document_string = f"Document {document.metadata['CELEX_ID']}:\n"
            document_string += f"{self.block_quotes}\n"
            document_string += document.page_content
            document_string += f"\n{self.block_quotes}"

            document_strings.append(document_string)

        return self.delimiter.join(document_strings)
