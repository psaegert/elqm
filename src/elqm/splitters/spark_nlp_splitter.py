import sparknlp
from pyspark.ml import Pipeline
from sparknlp.annotator import DocumentAssembler, DocumentCharacterTextSplitter


class SparkNLPSplitter():
    """
    For transforming the footnotes in the documents.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str]) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

        # Initialize SparkSession
        self.spark = sparknlp.start()

    def split_text(self, document_text: str) -> list[str]:
        """
        Uses SparkNLP to split the input string into chunks

        Args:
            document_text: The string, which will be chunked
        Returns:
            list[str]: List of strings (chunks)
        """

        # document_text = "This is a sample text that will be chunked Another piece of text to demonstrate chunking My friend was an enthusiastic musician, being himself not only"

        # For SparkNLP, we need to put document_text into a DF with a single column "text" and a single row
        document_text_df = self.spark.createDataFrame(
            [(document_text,)], ["text"])

        # We have to initialize the SparkNLP Pipeline objects "DocumentAssembler" and "DocumentCharacterTextSplitter"
        documentAssembler = DocumentAssembler().setInputCol("text")

        textSplitter = DocumentCharacterTextSplitter() \
            .setInputCols(["document"]) \
            .setOutputCol("splits") \
            .setSplitPatterns(self.separators) \
            .setChunkSize(self.chunk_size) \
            .setChunkOverlap(self.chunk_overlap) \
            .setPatternsAreRegex(False) \
            .setExplodeSplits(True)

        # Create Spark Pipeline with the initialized pipeline objects
        pipeline = Pipeline().setStages([documentAssembler, textSplitter])

        # Chunk the document_text and as output we get a DF with "result"-column (and 3 others for debugging)
        # , where each entry is a chunk
        document_text_chunked_df = pipeline.fit(
            document_text_df).transform(document_text_df)

        # Convert the entries in the "result"-column of the DF to a list of strings (where each
        # entry in the DF is a an entry in the list)
        document_text_chunked = document_text_chunked_df.select(
            "splits.result").rdd.flatMap(lambda x: x[0]).collect()

        return document_text_chunked
