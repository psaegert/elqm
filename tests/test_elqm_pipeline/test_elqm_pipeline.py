import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import pytest
from dynaconf import Dynaconf

from elqm import ELQMPipeline
from elqm.utils import cache_exists, clear_cache, get_dir

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestELQMPipeline(unittest.TestCase):
    def setUp(self) -> None:
        # Load the pytest config
        self.config = Dynaconf(settings_files=os.path.join(get_dir("configs"), 'pytest.yaml'))

        data_dir = get_dir("data", "pytest", create=True)  # Automatically creates the directory if it doesn't exist

        # Move the pytest data (../data/*.json) into the data dir (get_dir("data", )/pytest/*.json)
        for file in os.listdir(os.path.join(os.path.dirname(__file__), "..", "data")):
            shutil.copy(os.path.join(os.path.dirname(__file__), "..", "data", file), data_dir)

        self.mocked_model = MagicMock(invoke=MagicMock(return_value="Hello There!"))

    def tearDown(self) -> None:
        # Delete the pytest data dir
        shutil.rmtree(get_dir("data", "pytest"))
        clear_cache(index_name="pytest")

    @patch('elqm.utils.is_ollama_serve_running', return_value=True)
    @patch('elqm.factories.models.is_model_installed', return_value=True)
    def test_init_no_cache(self, mock_is_model_installed: MagicMock, mock_is_model_installed_2: MagicMock) -> None:
        clear_cache(index_name="pytest")
        self.assertFalse(cache_exists('pytest', retriever=None))

        elqm_pipeline = ELQMPipeline(self.config)
        self.assertIsNotNone(elqm_pipeline)

        self.assertTrue(cache_exists('pytest', retriever=None))

        clear_cache(index_name="pytest")

        self.assertFalse(cache_exists('pytest', retriever=None))

    @patch('elqm.utils.is_ollama_serve_running', return_value=True)
    @patch('elqm.factories.models.is_model_installed', return_value=True)
    def test_init_with_cache(self, mock_is_model_installed: MagicMock, mock_is_model_installed_2: MagicMock) -> None:
        clear_cache(index_name="pytest")
        self.assertFalse(cache_exists('pytest', retriever=None))

        elqm_pipeline = ELQMPipeline(self.config)

        self.assertTrue(cache_exists('pytest', retriever=None))

        del elqm_pipeline

        elqm_pipeline_2 = ELQMPipeline(self.config)

        self.assertIsNotNone(elqm_pipeline_2)

        clear_cache(index_name="pytest")

        self.assertFalse(cache_exists('pytest', retriever=None))

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Running LLMs on GitHub Actions would be too slow.")  # See test_answer_no_cache_mocked
    def test_answer_no_cache(self) -> None:
        clear_cache(index_name="pytest")
        self.assertFalse(cache_exists('pytest', retriever=None))

        elqm_pipeline = ELQMPipeline(self.config)
        answer = elqm_pipeline.answer("What is the meaning of life?")

        self.assertIsNotNone(answer)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

        clear_cache(index_name="pytest")

        self.assertFalse(cache_exists('pytest', retriever=None))

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Running LLMs on GitHub Actions would be too slow.")  # See test_answer_with_cache_mocked
    def test_answer_with_cache(self) -> None:
        clear_cache(index_name="pytest")
        self.assertFalse(cache_exists('pytest', retriever=None))

        elqm_pipeline = ELQMPipeline(self.config)

        self.assertTrue(cache_exists('pytest', retriever=None))

        del elqm_pipeline

        elqm_pipeline_2 = ELQMPipeline(self.config)

        answer = elqm_pipeline_2.answer("What is the meaning of life?")

        self.assertIsNotNone(answer)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

        clear_cache(index_name="pytest")

        self.assertFalse(cache_exists('pytest', retriever=None))

    @patch('elqm.utils.is_ollama_serve_running', return_value=True)
    @patch('elqm.factories.models.is_model_installed', return_value=True)
    def test_answer_no_cache_mocked(self, mock_is_model_installed: MagicMock, mock_is_ollama_serve_running: MagicMock) -> None:
        clear_cache(index_name="pytest")
        self.assertFalse(cache_exists('pytest', retriever=None))

        elqm_pipeline = ELQMPipeline(self.config)
        elqm_pipeline.model = self.mocked_model

        elqm_pipeline.answer("What is the meaning of life?")

        elqm_pipeline.model.invoke.assert_called_once()

        clear_cache(index_name="pytest")

        self.assertFalse(cache_exists('pytest', retriever=None))

    @patch('elqm.utils.is_ollama_serve_running', return_value=True)
    @patch('elqm.factories.models.is_model_installed', return_value=True)
    def test_answer_with_cache_mocked(self, mock_is_model_installed: MagicMock, mock_is_ollama_serve_running: MagicMock) -> None:
        clear_cache(index_name="pytest")
        self.assertFalse(cache_exists('pytest', retriever=None))

        elqm_pipeline = ELQMPipeline(self.config)
        elqm_pipeline.model = self.mocked_model

        self.assertTrue(cache_exists('pytest', retriever=None))

        del elqm_pipeline

        elqm_pipeline_2 = ELQMPipeline(self.config)
        elqm_pipeline_2.model = self.mocked_model

        answer = elqm_pipeline_2.answer("What is the meaning of life?")

        elqm_pipeline_2.model.invoke.assert_called_once()
        self.assertTrue(answer.startswith("Hello There!"))  # Do not check the sources

        clear_cache(index_name="pytest")

        self.assertFalse(cache_exists('pytest', retriever=None))

    @patch('elqm.utils.is_ollama_serve_running', return_value=True)
    @patch('elqm.factories.models.is_model_installed', return_value=True)
    def test_clear_chat_history(self, mock_is_model_installed: MagicMock, mock_is_model_installed_2: MagicMock) -> None:
        clear_cache(index_name="pytest")
        self.assertFalse(cache_exists('pytest', retriever=None))

        elqm_pipeline = ELQMPipeline(self.config)

        elqm_pipeline.clear_chat_history()

        self.assertEqual(elqm_pipeline.chat_history_memory.get_memory(), [])

        clear_cache(index_name="pytest")

        self.assertFalse(cache_exists('pytest', retriever=None))
