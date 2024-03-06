import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from elqm.utils import cache_exists, create_citation_link, deduplicate_retrieved_documents, extract_metadata, format_date_yyyy_mm_dd, get_dir, get_nested_value, get_raw_data, is_model_installed, is_ollama_serve_running

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestGetDirFunction(unittest.TestCase):

    def test_get_data_dir_with_valid_args(self) -> None:
        """Test if get_data_dir returns the correct path with valid arguments."""
        expected_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pytest_dir', 'test', 'dir'))
        actual_path = get_dir("pytest_dir", 'test', 'dir', create=True)
        self.assertEqual(expected_path, actual_path)

        shutil.rmtree(actual_path)
        shutil.rmtree(os.path.join(os.path.dirname(__file__), '..', 'pytest_dir'))

    def test_get_data_dir_creates_directory(self) -> None:
        """Test if get_data_dir creates the directory if it does not exist."""
        test_path = get_dir("pytest_dir", 'new_test_dir', create=True)
        self.assertTrue(os.path.isdir(test_path))

        # Clean up
        shutil.rmtree(os.path.join(os.path.dirname(__file__), '..', 'pytest_dir'))

    def test_get_data_dir_with_invalid_args(self) -> None:
        """Test if get_data_dir raises TypeError with non-string arguments."""
        with self.assertRaises(TypeError):
            get_dir("pytest_dir", 123, 'dir', create=True)

    def test_get_data_dir_with_no_args(self) -> None:
        """Test if get_data_dir works with no additional arguments."""
        expected_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pytest_dir'))
        actual_path = get_dir("pytest_dir", create=True)
        self.assertEqual(expected_path, actual_path)

        shutil.rmtree(actual_path)
        shutil.rmtree(os.path.join(os.path.dirname(__file__), '..', 'pytest_dir'), ignore_errors=True)


class TestGetRawDataFunction(unittest.TestCase):

    def test_get_raw_data_returns_dict(self) -> None:
        """Test if get_raw_data returns a dictionary."""
        # Create a data dir if it does not exist

        data_dir = get_dir("data", "pytest", create=True)

        raw_data = get_raw_data(data_dir)
        self.assertIsInstance(raw_data, dict)

        shutil.rmtree(get_dir("data", "pytest"))


class TestCacheFunctions(unittest.TestCase):

    def test_cache_exists_returns_false_for_nonexistent_cache(self) -> None:
        """Test if cache_exists returns False for a nonexistent cache."""
        cache_key = "nonexistent_cache"
        self.assertFalse(cache_exists(cache_key, retriever=None))


class TestOllamaReady(unittest.TestCase):

    @pytest.mark.skipif(not IN_GITHUB_ACTIONS, reason="Should not control local Ollama processes. Only test for negatives.")
    def test_is_ollama_serve_running(self) -> None:
        """Test if is_ollama_serve_running returns False when Ollama is not running."""
        self.assertFalse(is_ollama_serve_running())

    @patch('elqm.utils.is_ollama_serve_running', return_value=True)
    @pytest.mark.skipif(not IN_GITHUB_ACTIONS, reason="Should not control local Ollama processes. Only test for negatives.")
    def test_is_model_installed(self, mock_is_ollama_serve_running: MagicMock) -> None:
        """Test if is_model_installed returns False when Ollama is not running."""
        self.assertFalse(is_model_installed("this_model_is_not_installed"))


class TestCreateCitationLink(unittest.TestCase):

    def test_create_citation_link(self) -> None:
        """Test if create_citation_link returns the correct link."""
        celex_id = "31983H0230"
        expected_link = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex_id}"
        actual_link = create_citation_link(celex_id)
        self.assertEqual(expected_link, actual_link)


class TestExtractMetadataFunction(unittest.TestCase):

    def test_extract_metadata(self) -> None:
        """Test if extract_metadata returns the correct metadata."""
        record = {
            "html": "Some HTML"
        }

        metadata = {
            "html": "Some HTML",
            "date_key": "2021-01-01",
        }

        new_metadata = extract_metadata(record, metadata, mapping={"date": {"type": "date", "json_key": "date_key", "description": "Date"}})

        self.assertTrue("html" in new_metadata)
        self.assertTrue("date" in new_metadata)

    def test_extract_nested_metadata(self) -> None:
        """Test if extract_metadata returns the correct metadata."""
        record = {
            "html": "Some HTML",
        }

        metadata = {
            "html": "Some HTML",
            "dates": {
                "date_key": "2021-01-01",
            }
        }

        new_metadata = extract_metadata(record, metadata, mapping={"date": {"type": "date", "json_key": ["dates", "date_key"], "description": "Date"}})

        self.assertTrue("html" in new_metadata)
        self.assertTrue("date" in new_metadata)


class TestFormatDateFunction(unittest.TestCase):

    def test_format_date_yyyy_mm_dd(self) -> None:
        """Test if format_date_yyyy_mm_dd returns the correct date."""
        date = "31/04/2021"
        expected_date = "2021-04-31"
        actual_date = format_date_yyyy_mm_dd(date)
        self.assertEqual(expected_date, actual_date)


class TestGetNestedValueFunction(unittest.TestCase):

    def test_get_nested_value(self) -> None:
        """Test if get_nested_value returns the correct value."""
        data = {
            "nested": {
                "key": "value"
            }
        }

        expected_value = "value"
        actual_value = get_nested_value(data, ["nested", "key"])
        self.assertEqual(expected_value, actual_value)

    def test_get_nested_value_returns_none(self) -> None:
        """Test if get_nested_value returns None for a nonexistent key."""
        data = {
            "nested": {
                "key": "value"
            }
        }

        actual_value = get_nested_value(data, ["nested", "nonexistent_key"])
        self.assertEqual("", actual_value)


class TestRemoveDuplicatesOrderedFunction(unittest.TestCase):

    def test_remove_duplicates_ordered(self) -> None:
        """Test if remove_duplicates_ordered returns the correct list of documents."""
        documents = [
            Document(page_content="Page 1", metadata={"CELEX_ID": "31983H0230"}),
            Document(page_content="Page 2", metadata={"CELEX_ID": "31983H0230"}),
            Document(page_content="Page 3", metadata={"CELEX_ID": "a"}),
            Document(page_content="Page 1", metadata={"CELEX_ID": "31983H0230"}),
        ]

        expected_documents = [
            Document(page_content="Page 1", metadata={"CELEX_ID": "31983H0230"}),
            Document(page_content="Page 2", metadata={"CELEX_ID": "31983H0230"}),
            Document(page_content="Page 3", metadata={"CELEX_ID": "a"}),
        ]

        actual_documents = deduplicate_retrieved_documents(documents)

        self.assertEqual(expected_documents, actual_documents)
