import re
import textwrap

import numpy as np
from langchain.llms.ollama import Ollama
from langchain_core.documents import Document
from tqdm import tqdm

QUSTION_TYPES = {
    "confirmation": "Focus only on confirmation questions, i.e. questions that can be answered with a yes or no.",
    "factoid": "Focus only on factoid questions, that usually begin with a who, what, where, when, why, or how.",
    "list": "Focus only on list questions, i.e. questions that are answered with a list of items.",
    "causal": "Focus only on causal questions, i.e. questions that begin with why or how.",
    "hypothetical": "Focus only on hypothetical questions, i.e. questions that ask what if.",
    "complex": "Focus only on complex questions, i.e. questions that require multi-step reasoning and comparisons.",
    "default": ""
}


def generate_question_answer_pairs(context: str, prompt: str | None = None, question_type: str | None = None, model: str = 'llama2', n: int = 1, verbose: bool = False) -> list[tuple[str, str]]:
    """
    Generate question-answer pairs from a given context using the Ollama model.

    Parameters
    ----------
    context : str
        The context from which to generate the question-answer pairs.
    prompt : str, optional
        The prompt to use for the Ollama model. If None, a default prompt is used.
    question_type : str, optional
        The type of questions to generate. If None, any type of question is generated.
    model : str, optional
        The model to use for generating the question-answer pairs, by default 'llama2'
    n : int, optional
        The number of question-answer pairs to generate.
    verbose : bool, optional
        Whether to print the output of the Ollama model.

    Returns
    -------
    list[tuple[str, str]]
        A list of question-answer pairs.
    """
    if prompt is None:

        question_type_prompt = QUSTION_TYPES.get(question_type, QUSTION_TYPES["default"])  # type: ignore

        # FIXME: Rerun with correct spelling of excerpt
        prompt = textwrap.dedent(f'''You are an oracle system for a Retrieval Augmented Generation System, that guesses questions that would be answered by a particular exerpt of text.
            Given the following exerpt of text, generate {n} question{"s" if n > 1 else ""} that can be answered by the exerpt of text.
            ```
            {context}
            ```
            {question_type_prompt}
            Format the pairs as follows:
            ```
            {{QUESTION i}}: <question i> {{ANSWER i}}: <answer i>
            ```
            Do not add any additional newlines between the pairs. Directly continue with the answer after the question.
            Only add a newline between the pairs (after the answer) if you want to add more pairs.
            Do not deviate from this format, since it will be used to extract the questions with the following regex: `{{QUESTION \\d+}}: .+ {{ANSWER \\d+}}: .+`
            ''')

    # Clear the message history by initializing a new Ollama instance
    ollama = Ollama(
        base_url="http://localhost:11434",
        model=model,
        verbose=True,
        stop=["<|im_end|>"]
    )

    if verbose:
        # Stream the output
        response = ""
        for token in ollama.stream(prompt):
            response += token
            print(token, end="")
    else:
        # Generate the question-answer pairs
        response = ollama.invoke(prompt)

    # Filter out the question-answer pairs
    qa_pairs = re.findall(r"\{QUESTION (\d+)\}:\s*(.*?)\s*\{ANSWER \1\}:\s*(.*?)\s*(?=\{QUESTION \d+\}|$)", response)
    qa_pairs = [(q.strip(), a.strip()) for _, q, a in qa_pairs if q.strip() and a.strip()]

    # Check if any question-answer pairs were generated
    if len(qa_pairs) == 0 and verbose:
        print(f"No question-answer pairs were generated: {qa_pairs}")

    return qa_pairs


def generate_oracle_dataset(documents: list[Document], model: str = 'llama2', question_type: str | list[str] | None = None, n_questions_per_type: int = 1, strategy: str = "all", random_seed: int | None = None, verbose: bool = False) -> list[Document]:
    """
    Generate a dataset of question-answer pairs from a given directory of data.

    Parameters
    ----------
    documents : list[Document]
        The list of documents to generate question-answer pairs from.
    model : str, optional
        The model to use for generating the question-answer pairs, by default 'llama2'
    question_type : str | list[str], optional
        The type of question to generate. If None, all types are used, by default None
    n_questions_per_type : int, optional
        The number of questions to generate per question type, by default 1
    strategy : str, optional
        The strategy for choosing the question type per document, by default "all". One of "all" or "random".
    verbose : bool, optional
        Whether to print progress, by default False
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if isinstance(question_type, str):
        question_type = [question_type]
    elif question_type is None:
        question_type = list(QUSTION_TYPES.keys())

    # Generate question-answer pairs for each document
    for document in tqdm(documents, desc="Generating question-answer pairs", disable=not verbose):

        # If a raw text is available (as in the case of enrichment), use the raw text stored in the metadata
        if 'text' in document.metadata:
            context = document.metadata['text']
        # Otherwise, use the page content
        else:
            context = document.page_content

        document.metadata['oracle_pairs'] = []

        if strategy == "all":
            for qt in question_type:
                pairs = generate_question_answer_pairs(context, question_type=qt, model=model, n=n_questions_per_type, verbose=False)
                for pair in pairs:
                    document.metadata['oracle_pairs'].append({'question': pair[0], 'answer': pair[1], 'type': qt})
        elif strategy == "random":
            qt = np.random.choice(question_type)
            pairs = generate_question_answer_pairs(context, question_type=qt, model=model, n=n_questions_per_type, verbose=False)
            for pair in pairs:
                document.metadata['oracle_pairs'].append({'question': pair[0], 'answer': pair[1], 'type': qt})

    return documents
