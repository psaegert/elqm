import os
import shutil
import textwrap
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from dynaconf import Dynaconf

from elqm import ELQMPipeline
from elqm.eval import Evaluation
from elqm.utils import clear_cache, get_dir

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluation = Evaluation()

    def test_retriever_recall(self) -> None:
        self.assertEqual(self.evaluation.scores_functions['RET_RC'](references=['1', '2'], candidates=['1', '2'], max_k=2), [0.5, 1.0])
        self.assertEqual(self.evaluation.scores_functions['RET_RC'](references=['1', '2'], candidates=['1']), [0.5])
        self.assertEqual(self.evaluation.scores_functions['RET_RC'](references=['1', '2'], candidates=['1', '3']), [0.5])
        self.assertEqual(self.evaluation.scores_functions['RET_RC'](references=['1', '2'], candidates=['3']), [0.0])

    def test_retriever_precision(self) -> None:
        self.assertEqual(self.evaluation.scores_functions['RET_PR'](references=['1', '2'], candidates=['1', '2']), [1.0])
        self.assertEqual(self.evaluation.scores_functions['RET_PR'](references=['1', '2'], candidates=['1']), [1.0])
        self.assertEqual(self.evaluation.scores_functions['RET_PR'](references=['1', '2'], candidates=['1', '3'], max_k=2), [1.0, 0.5])
        self.assertEqual(self.evaluation.scores_functions['RET_PR'](references=['1', '2'], candidates=['3']), [0.0])

    def test_retriever_f1(self) -> None:
        self.assertEqual(self.evaluation.scores_functions['RET_F1'](references=['1', '2'], candidates=['1', '2'], max_k=2), [0.6666666666666666, 1.0])
        self.assertEqual(self.evaluation.scores_functions['RET_F1'](references=['1', '2'], candidates=['1']), [0.6666666666666666])
        self.assertEqual(self.evaluation.scores_functions['RET_F1'](references=['1', '2'], candidates=['1', '3'], max_k=2), [0.6666666666666666, 0.5])
        self.assertEqual(self.evaluation.scores_functions['RET_F1'](references=['1', '2'], candidates=['3']), [0.0])

    def test_bleu_match(self) -> None:
        result = self.evaluation.scores_functions['A_BL'](predictions=['Hello world'], references=[['Hello world']])

        self.assertEqual(type(result), dict)
        self.assertEqual(result['precisions'], [1.0, 1.0, 0, 0])

    def test_bleu_mismatch(self) -> None:
        result = self.evaluation.scores_functions['A_BL'](predictions=['Hello world'], references=[['Hello']])

        self.assertEqual(type(result), dict)
        self.assertEqual(result['precisions'], [0.5, 0, 0, 0])

    def test_rouge_match(self) -> None:
        result = self.evaluation.scores_functions['A_RG'](predictions=['Hello world'], references=[['Hello world']])

        self.assertEqual(type(result), dict)
        self.assertEqual(result['rouge1'], 1)

    def test_rouge_mismatch(self) -> None:
        result = self.evaluation.scores_functions['A_RG'](predictions=['Hello world'], references=[['Hello']])

        self.assertEqual(type(result), dict)
        self.assertEqual(result['rouge1'], 2 / 3)

    def test_bertscore_match(self) -> None:
        result = self.evaluation.scores_functions['A_BERT_RAG'](predictions=['Hello world'], references=[['Hello world']])

        self.assertEqual(type(result), dict)
        self.assertTrue(result['precision'][0] > 0.99)

    def test_bertscore_mismatch(self) -> None:
        result = self.evaluation.scores_functions['A_BERT_RAG'](predictions=['Hello world'], references=[['Hello']])

        self.assertEqual(type(result), dict)
        self.assertLess(result['precision'][0], 0.9)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Running LLMs on GitHub Actions would be too slow.")
    def test_answer_relevance_relevant(self) -> None:
        question = "Is the capital of France Paris?"

        answer = textwrap.dedent("""
        Yes, the capital of France is Paris.
        Paris is located in northern central France on the river Seine.
        It has been the capital city of France since the late 5th century, though some earlier French kings had their main residence in other cities such as Laon and Soissons.
        """)

        oracle_qa_pairs, relevance_scores = self.evaluation.scores_functions['A_AR'](
            question=question,
            answer=answer,
            n_questions=2,
            similarity=lambda a, b: self.evaluation.scores_functions['A_BERT_RAG'](predictions=[a], references=[b]))

        self.assertTrue(isinstance(oracle_qa_pairs, list))
        self.assertTrue(isinstance(oracle_qa_pairs[0], tuple))
        self.assertTrue(isinstance(oracle_qa_pairs[0][0], str))

        self.assertTrue(isinstance(relevance_scores, dict))
        self.assertGreaterEqual(relevance_scores['precision'], 0.80)  # FIXME: Rather low due to inteterministic question generation
        self.assertGreaterEqual(relevance_scores['recall'], 0.80)
        self.assertGreaterEqual(relevance_scores['f1'], 0.80)


class TestEvaluate(unittest.TestCase):
    def setUp(self) -> None:
        # Load the pytest config
        self.config = Dynaconf(settings_files=os.path.join(get_dir("configs"), 'pytest.yaml'))

        data_dir = get_dir("data", "pytest", create=True)  # Automatically creates the directory if it doesn't exist

        # Move the pytest data (./data/*.json) into the data dir (get_dir("data", )/pytest/*.json)
        for file in os.listdir(os.path.join(os.path.dirname(__file__), "..", "data")):
            shutil.copy(os.path.join(os.path.dirname(__file__), "..", "data", file), data_dir)

        self.mocked_model = MagicMock(invoke=MagicMock(return_value="Hello There!"))

        self.oracle_data = pd.DataFrame({
            'question': ['What is the capital of Germany?', 'What is the capital of France?'],
            'answer': ['Berlin', 'Paris'],
            'source': ['1', '2']
        })

    def tearDown(self) -> None:
        # Delete the pytest data dir
        shutil.rmtree(get_dir("data", "pytest"))

        clear_cache(index_name="pytest")

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Running LLMs on GitHub Actions would be too slow.")  # See test_answer_no_cache_mocked
    @patch('elqm.factories.models.is_model_installed', return_value=True)
    def test_evaluate(self, mock_is_model_installed: MagicMock) -> None:
        self.pipeline = ELQMPipeline(config=self.config)
        self.pipeline.model = self.mocked_model

        eval = Evaluation()

        results = eval.evaluate(self.pipeline, self.oracle_data)

        self.assertIsInstance(results, dict)
