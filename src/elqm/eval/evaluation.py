from typing import Callable

import evaluate
import numpy as np
import pandas as pd
from numpy.core._exceptions import UFuncTypeError
from tqdm import tqdm

from elqm import ELQMPipeline
from elqm.eval.oracle import generate_question_answer_pairs


class Evaluation():
    def __init__(self) -> None:
        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load('bertscore')

        self.scores_functions = {
            'A_BL': self.bleu.compute,
            'A_RG': self.rouge.compute,
            'A_BERT_RAG': lambda predictions, references: self.bertscore.compute(predictions=predictions, references=references, lang='en'),
            'A_BERT': lambda predictions, references: self.bertscore.compute(predictions=predictions, references=references, lang='en'),
            'RET_RC': retriever_recall,
            'RET_PR': retriever_precision,
            'RET_F1': retriever_f1,
            'RET_RR': retriever_reciprocal_rank,
            'RET_AP': retriever_average_precision,
            'RET_CG': cumulative_gain,
            'RET_DCG': discounted_cumulative_gain,
            'RET_NDCG': normalized_discounted_cumulative_gain,
            'A_AR': self.answer_relevance
        }

    def evaluate(self, elqm: ELQMPipeline, dataset: pd.DataFrame) -> dict[str, list]:
        """
        Evaluates the pipeline on the given dataset by comparing retrieved documents to ground truth documents and answers to ground truth answers.

        Parameters
        ----------
        elqm : ELQMPipeline
            The pipeline to evaluate.
        dataset : pd.DataFrame
            The dataset to evaluate on with keys 'question', 'answer' and 'source'.

        Returns
        -------
        dict[str, list]
            A dictionary of scores for each metric.
        """
        results: dict[str, list] = {name: [] for name in self.scores_functions.keys()}

        if elqm.config['retriever'] == 'ensemble':
            max_k = max(retriever['retriever_args']['k_retrieved_documents'] for retriever in elqm.config["retriever_args"]["retrievers"])
        else:
            max_k = elqm.config["retriever_args"]['k_retrieved_documents']

        for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc='Evaluating'):
            elqm.clear_chat_history()
            output = elqm.invoke(row['question'])

            prompt_no_rag = elqm.prompt.invoke({"question": row['question'], "context": "", "chat_history": []})
            model_outout_no_rag = elqm.model.invoke(input=prompt_no_rag)
            parsed_model_outout_no_rag = elqm.output_parser.invoke(model_outout_no_rag)

            # Retrieved documents
            candidate_document_ids = [d.metadata['ID'] for d in output['retrieved_documents']]

            reference_document_id = row['source']

            results['RET_RC'].append(retriever_recall(references=[reference_document_id], candidates=candidate_document_ids, max_k=max_k))
            results['RET_PR'].append(retriever_precision(references=[reference_document_id], candidates=candidate_document_ids, max_k=max_k))
            results['RET_F1'].append(retriever_f1(references=[reference_document_id], candidates=candidate_document_ids, max_k=max_k))
            results['RET_RR'].append(retriever_reciprocal_rank(references=[reference_document_id], candidates=candidate_document_ids))
            results['RET_AP'].append(retriever_average_precision(references=[reference_document_id], candidates=candidate_document_ids, max_k=max_k))
            results['RET_CG'].append(cumulative_gain(references=[reference_document_id], candidates=candidate_document_ids, max_k=max_k))
            results['RET_DCG'].append(discounted_cumulative_gain(references=[reference_document_id], candidates=candidate_document_ids, max_k=max_k))
            results['RET_NDCG'].append(normalized_discounted_cumulative_gain(references=[reference_document_id], candidates=candidate_document_ids, max_k=max_k))

            # Answers
            candidate_answer = output['parsed_model_output']
            reference_answer = row['answer']

            results['A_BL'].append(self.scores_functions['A_BL'](
                predictions=[candidate_answer],
                references=[[reference_answer]]))
            results['A_RG'].append(self.scores_functions['A_RG'](
                predictions=[candidate_answer],
                references=[[reference_answer]]))
            results['A_BERT_RAG'].append(self.scores_functions['A_BERT_RAG'](
                predictions=[candidate_answer],
                references=[[reference_answer]]))
            results['A_BERT'].append(self.scores_functions['A_BERT'](
                predictions=[parsed_model_outout_no_rag],
                references=[[reference_answer]]))
            results['A_AR'].append(self.scores_functions['A_AR'](
                question=row['question'],
                answer=candidate_answer,
                n_questions=2,
                model=elqm.config.model if elqm.config.model else 'llama2',
                similarity=lambda a, b: self.scores_functions['A_BERT_RAG'](predictions=[a], references=[b])))

        return {k: v for k, v in results.items() if len(v) > 0}

    def answer_relevance(self, question: str, answer: str, n_questions: int = 1, model: str = 'llama2', similarity: Callable[[str, str], dict[str, list[float]]] | None = None) -> tuple[list[tuple[str, str]], dict[str, float]]:
        """
        Computes the relevance of the predicted answers to the reference answers.

        The answer a(q) for a given query q is considered relevant if potential questions
        q_i(a(q)) are similar to q.

        AR = 1/n * sum_i=1^n(sim(q, q_i(a(q))))

        where sim is the similarity function, q_i(a(q)) is the potential question for a given answer a(q),
        and n is the number of potential questions.

        Parameters
        ----------
        question : str
            The query.
        answer : str
            The predicted answer.
        n_questions : int, optional
            The number of potential questions, by default 1.
        model : str, optional
            The model to use for generating the potential questions, by default 'llama2'.
        similarity : Callable[[str, str], dict[str, list[float]]] | None, optional
            The similarity function to use, by default None. If None, uses the BERTScore similarity function.

        Returns
        -------
        list[tuple[str, str]]
            The potential question-answer pairs generated from the answer.
        dict[str, float]
            The relevance scores for each similarity metric.
        """
        if similarity is None:
            similarity = self.scores_functions['A_BERT_RAG']

        # For each answer, generate questions
        oracle_qa_pairs = generate_question_answer_pairs(
            context=answer,
            prompt=None,  # default
            question_type=None,  # default
            model=model,
            n=n_questions)

        # Compute the similarity score between the actual question and the generated questions from the answer
        similarity_dicts = [similarity(question, oracle_question) for oracle_question, _ in oracle_qa_pairs]

        # Average the similarity scores
        relevance_scores = {}

        if len(similarity_dicts) == 0:
            return oracle_qa_pairs, {'precision': 0, 'recall': 0, 'f1': 0}

        for metric in similarity_dicts[0].keys():
            try:
                relevance_scores[metric] = np.mean([similarity_dict[metric] for similarity_dict in similarity_dicts])
            # If np.mean fails (can happen if the values are strings), set the score to nan
            except UFuncTypeError:
                relevance_scores[metric] = np.nan

        return oracle_qa_pairs, relevance_scores


def retriever_reciprocal_rank(references: list[str], candidates: list[str]) -> float:
    """
    Computes the reciprocal rank of first relevant document in the candidates.

    Parameters
    ----------
    references : list[str]
        The reference document IDs considered relevant.
    candidates : list[str]
        The candidate document IDs retrieved by the system, in ranked order. Assumes that the candidates are sorted by relevance.

    Returns
    -------
    float
        The reciprocal rank.
    """
    for i, candidate in enumerate(candidates):
        if candidate in references:
            return 1 / (i + 1)

    return 0.0


def retriever_recall(references: list[str], candidates: list[str], max_k: int = 1) -> list[float]:
    """
    Computes the recall of the retriever from reference ids and candidate ids.

    Parameters
    ----------
    references : list[str]
        The reference document IDs considered relevant.
    candidates : list[str]
        The candidate document IDs retrieved by the system, in ranked order. Assumes that the candidates are sorted by relevance.
    max_k : int, optional
        The maximum number of retrieved documents to consider, by default 1.

    Returns
    -------
    list[float]
        The recall scores @k for k in [1, max_k].
    """
    if len(references) == 0:
        return [0.0] * max_k

    return [len(set(references).intersection(set(candidates[:k]))) / len(references) for k in range(1, max_k + 1)]


def retriever_precision(references: list[str], candidates: list[str], max_k: int = 1) -> list[float]:
    """
    Computes the precision of the retriever from reference ids and candidate ids.

    Parameters
    ----------
    references : list[str]
        The reference document IDs considered relevant.
    candidates : list[str]
        The candidate document IDs retrieved by the system, in ranked order. Assumes that the candidates are sorted by relevance.
    max_k : int, optional
        The maximum number of retrieved documents to consider, by default 1.

    Returns
    -------
    list[float]
        The precision scores @k for k in [1, max_k].
    """
    precisions: list[float] = []

    for k in range(1, max_k + 1):
        if len(candidates[:k]) == 0:
            precisions.append(0.0)
        else:
            precisions.append(len(set(references).intersection(set(candidates[:k]))) / len(candidates[:k]))

    return precisions


def retriever_f1(references: list[str], candidates: list[str], max_k: int = 1) -> list[float]:
    """
    Computes the f1 of the retriever from reference ids and candidate ids.

    Parameters
    ----------
    references : list[str]
        The reference document IDs considered relevant.
    candidates : list[str]
        The candidate document IDs retrieved by the system, in ranked order.
    max_k : int, optional
        The maximum number of retrieved documents to consider, by default 1.

    Returns
    -------
    list[float]
        The f1 scores @k for k in [1, max_k].
    """
    precisions = retriever_precision(references, candidates, max_k)
    recalls = retriever_recall(references, candidates, max_k)

    f1s: list[float] = []

    for precision, recall in zip(precisions, recalls):
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * (precision * recall) / (precision + recall))

    return f1s


def retriever_average_precision(references: list[str], candidates: list[str], max_k: int = 1) -> list[float]:
    """
    Computes the average precision of the retriever from reference ids and candidate ids for each k in 1 to max_k.

    Parameters
    ----------
    references : list[str]
        The reference document IDs considered relevant.
    candidates : list[str]
        The candidate document IDs retrieved by the system, in ranked order.
    max_k : int
        The maximum number of retrieved documents to consider.

    Returns
    -------
    list[float]
        The list of average precision scores @k for k in [1, max_k].
    """
    if len(references) == 0 or max_k < 1:
        return [0.0] * max_k

    hits = np.zeros(max(max_k, len(candidates)), dtype=bool)
    hits[:len(candidates)] = [True if candidate in references else False for candidate in candidates]

    precisions = np.array(retriever_precision(references, candidates, max_k))

    average_precisions = [np.sum(precisions[:k] * hits[:k]) / len(references) for k in range(1, max_k + 1)]

    return average_precisions


def cumulative_gain(references: list[str], candidates: list[str], max_k: int = 1) -> float:
    """
    Computes the cumulative gain of the retriever from reference ids and candidate ids.

    Parameters
    ----------
    references : list[str]
        The reference document IDs considered relevant.
    candidates : list[str]
        The candidate document IDs retrieved by the system, in ranked order.
    max_k : int, optional
        The maximum number of retrieved documents to consider. If None, considers all documents.

    Returns
    -------
    float
        The cumulative gain for the given set of references and candidates.
    """
    return sum([1 if candidate in references else 0 for candidate in candidates[:max_k]])


def discounted_cumulative_gain(references: list[str], candidates: list[str], max_k: int = 1) -> float:
    """
    Computes the discounted cumulative gain of the retriever from reference ids and candidate ids.

    Parameters
    ----------
    references : list[str]
        The reference document IDs considered relevant.
    candidates : list[str]
        The candidate document IDs retrieved by the system, in ranked order.
    max_k : int, optional
        The maximum number of retrieved documents to consider. If None, considers all documents.

    Returns
    -------
    float
        The discounted cumulative gain for the given set of references and candidates.
    """
    relevance_mask = [1 if candidate in references else 0 for candidate in candidates[:max_k]]

    # Pad the relevance mask with zeros if the number of candidates is less than max_k
    relevance_mask += [0] * max(0, max_k - len(relevance_mask))

    return sum([relevance_mask[i] / np.log2(i + 2) for i in range(max_k)])  # + 2 since i starts at 0


def ideal_discounted_cumulative_gain(references: list[str], candidates: list[str]) -> float:
    """
    Computes the ideal discounted cumulative gain of the retriever from reference ids and candidate ids.

    Parameters
    ----------
    references : list[str]
        The reference document IDs considered relevant.
    candidates : list[str]
        The candidate document IDs retrieved by the system, in ranked order.

    Returns
    -------
    float
        The ideal discounted cumulative gain for the given set of references and candidates.
    """
    return discounted_cumulative_gain(references, candidates, max_k=len(references))  # Only sum until the number of relevant documents (ideal scenario)


def normalized_discounted_cumulative_gain(references: list[str], candidates: list[str], max_k: int = 1) -> float:
    """
    Computes the normalized discounted cumulative gain of the retriever from reference ids and candidate ids.

    Parameters
    ----------
    references : list[str]
        The reference document IDs considered relevant.
    candidates : list[str]
        The candidate document IDs retrieved by the system, in ranked order.
    max_k : int, optional
        The maximum number of retrieved documents to consider. If None, considers all documents.

    Returns
    -------
    float
        The normalized discounted cumulative gain for the given set of references and candidates.
    """
    dcg = discounted_cumulative_gain(references, candidates, max_k)
    idcg = ideal_discounted_cumulative_gain(references, candidates)

    if idcg == 0 or dcg == 0:
        return 0.0

    return dcg / idcg
