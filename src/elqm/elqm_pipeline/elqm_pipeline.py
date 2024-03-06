import os
import textwrap
import warnings

import langchain
from dynaconf import Dynaconf

from elqm.factories import DocumentLoaderFactory, EmbeddingFactory, ModelFactory, OutputParserFactory, PostprocessorFactory, PromptFactory, RetrieverFactory
from elqm.logging.conversation_logger import ConversationLogger
from elqm.memory.chat_history_memory import ChatHistoryMemory
from elqm.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from elqm.utils import create_citation_link, deduplicate_retrieved_documents, get_dir, is_config_cached, is_model_installed, is_ollama_serve_running


class ELQMPipeline:
    """
    The ELQM Pipeline, a RAG-based Question Answering Pipeline for legal documents.

    Attributes
    ----------
    config : Dynaconf
        The configuration.
    config_name : str
        The name of the config file.
    preprocessing_pipeline : PreprocessingPipeline
        The preprocessing pipeline.
    loader : DocumentLoader
        The document loader.
    splitter : Splitter
        The document splitter.
    embeddings : Embeddings
        The embeddings used for the index and retrieval.
    vectorstore: VectorStore
        The vectorstore used for the index and retrieval.
    retriever : Retriever
        The retriever.
    postprocessor : Postprocessor
        The postprocessor used to postprocess the retrieved documents before feeding them to the model.
    prompt : Prompt
        The prompt used to generate the input for the model.
    model : BaseLLM
        The model used to generate the answer.
    output_parser : OutputParser
        The output parser used to parse the model output.
    chat_history_memory : ChatHistoryMemory
        The chat history memory used to store the chat history.
    debug : bool
        Whether to run in debug mode.
    log_conversation : bool
        Whether to log the conversation.
    logger : ConversationLogger
        The conversation logger.
    """
    def __init__(self, config: Dynaconf):
        if not is_ollama_serve_running():
            warnings.warn("\n\033[93mollama-serve is not running. Please run `ollama serve` to start it.\033[0m")

        if not is_model_installed(config.model):
            warnings.warn(f"\n\033[93mModel {config.model} is not installed. Install it with `ollama pull {config.model}` or check for models on https://ollama.ai/library?sort=newest\033[0m")

        # Check if a cache with the filename of the config exists
        self.config = config
        self.config_name = os.path.splitext(os.path.basename(config.settings_file))[0]

        self.preprocessing_pipeline = PreprocessingPipeline(config.preprocessing_steps, config.debug)

        self.loader = DocumentLoaderFactory.get_document_loader(
            document_loader_name=config.document_loader,
            **{k.lower(): v for k, v in dict(config).items()})

        print(f'Checking if cache exists and complete for config {self.config_name}')
        if not is_config_cached(config) or not config.cache:
            if not is_config_cached(config):
                print(f'Cache not found or incomplete for {config.index_name}')
            else:
                print(f'Cache disabled for {config.index_name}')

            # When trying to use the cache but it doesn't exist yet, check if the preprocessed_documents directory exists and is not empty to load the preprocessed documents
            # HACK: Does not check for completeness
            if config.cache and os.path.exists(get_dir("cache", config.index_name, "preprocessed_documents")) and os.listdir(get_dir("cache", config.index_name, "preprocessed_documents")):
                print('Loading cached preprocessed documents.')
            # Preprocess Documents
            else:
                print('Preprocessing documents:')
                document_dict = self.preprocessing_pipeline.load_documents(os.path.join(get_dir("data"), config.data_dir), verbose=True)
                document_dict = self.preprocessing_pipeline.preprocess(document_dict, verbose=True)
                self.preprocessing_pipeline.save_documents(
                    document_dict,
                    get_dir("cache", config.index_name, "preprocessed_documents", create=True),
                    drop_keys=config.preprocessing_args.drop_keys, verbose=True)

            # Load Documents
            print("Loading documents:")
            chunks = self.loader.load()
            print(f'Created Document Loader {self.loader.__class__.__name__}')
            print(f'Loaded {len(chunks)} documents')

        else:
            print(f'Using cache {config.index_name} for config {self.config_name}')
            chunks = None

        # Get an Embedding
        self.embeddings = EmbeddingFactory.get_embedding(
            config.embeddings,
            embedding_kwargs=config.embedding_args)
        print(f'Created Embedding {self.embeddings.__class__.__name__}')

        # Set up a document retriever
        self.retriever = RetrieverFactory.get_retriever(
            retriever_name=config.retriever,
            embeddings_object=self.embeddings,
            documents=chunks,
            **{k.lower(): v for k, v in dict(config).items()})
        print(f'Created Retriever {self.retriever.__class__.__name__}')

        # Set up a retrieved documents postprocessor
        self.postprocessor = PostprocessorFactory.get_postprocessor(
            postprocessing_config_name=config.postprocessor,
            **{k.lower(): v for k, v in dict(config.postprocessor_args).items()})

        # Get a prompt, model and output parser
        self.prompt = PromptFactory.get_prompt(config.prompt)
        self.model, self.model_max_length = ModelFactory.get_model(config.model)
        self.output_parser = OutputParserFactory.get_output_parser(config.output_parser)

        # Set up the chat history memory
        self.chat_history_memory = ChatHistoryMemory(
            history_size=config.chat_history_window_size,
            max_length=self.model_max_length)

        self.debug = config.debug
        langchain.debug = self.debug

        self.log_conversation = config.log_conversation
        if self.log_conversation:
            self.logger = ConversationLogger(self.config_name)

    def __str__(self) -> str:
        return textwrap.dedent(f"""ELQM Pipeline:
            Preprocessing Pipeline: {self.preprocessing_pipeline}
            Document Loader: {self.loader}
            Embeddings: {self.embeddings}
            Retriever: {self.retriever}
            Postprocessor: {self.postprocessor}
            Prompt: {self.prompt}
            Model: {self.model}
            Output Parser: {self.output_parser}
            Chat History Memory: {self.chat_history_memory}
            Debug: {self.debug}
            Log Conversation: {self.log_conversation}
            """)

    def invoke(self, question: str) -> dict:
        """
        Run the pipeline for a question.

        Parameters
        ----------
        question : str
            The question to answer.

        Returns
        -------
        dict
            The output of the pipeline including
            - retrieved_documents: The retrieved documents.
            - prompt: The prompt used to generate the input for the model.
            - parsed_model_output: The parsed model output.
        """
        retrieved_documents = deduplicate_retrieved_documents(self.retriever.invoke(question))

        postprocessed_retrieved_documents = self.postprocessor.postprocess(retrieved_documents)

        chat_history = self.chat_history_memory.get_memory()

        prompt = self.prompt.invoke({
            "question": question,
            "context": postprocessed_retrieved_documents,
            "chat_history": chat_history,
        })

        model_outout = self.model.invoke(input=prompt)

        parsed_model_output = self.output_parser.invoke(model_outout)

        self.chat_history_memory.add_user_message(question)
        self.chat_history_memory.add_ai_message(model_outout)

        self.logger.log("user", question)
        self.logger.log("assistant", model_outout)

        return {
            "retrieved_documents": retrieved_documents,
            "prompt": prompt,
            "parsed_model_output": parsed_model_output,
        }

    def answer(self, question: str, chat_history: dict = None) -> str:
        """
        Answer a question with the pipeline and cite the sources.

        Parameters
        ----------
        question : str
            The question to answer.
        chat_history : dict, optional
            The chat history to use, by default None. This parameter exists for compatibility with the Gradio chat interface.

        Returns
        -------
        str
            The answer to the question.
        """
        result = self.invoke(question)

        answer = result['parsed_model_output']

        # HACK: Add a step in the pipeline that postprocesses the answer to add citations
        sources = [doc.metadata['CELEX_ID'] for doc in result['retrieved_documents']]
        unique_sources = set([source.split('_')[0] for source in sources if source in answer])

        if len(unique_sources) > 0:
            answer += "\n\nSources:"
            for source in unique_sources:
                if source in answer:
                    answer += f"\n- {create_citation_link(source.split('_')[0])}"  # Remove the ID added by the splitting

        return answer

    def clear_chat_history(self) -> None:
        """
        Clear the chat history.
        """
        self.chat_history_memory.clear_memory()
