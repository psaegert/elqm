import unittest
from unittest.mock import Mock

from elqm.postprocessing import AppendAllCitationPostprocessor


class TestAppendAllCitationPostprocessor(unittest.TestCase):

    def test_append_all_citation_postprocessor(self) -> None:
        """Test if the AppendAllCitationPostprocessor works correctly."""
        p = AppendAllCitationPostprocessor()
        objects = [
            Mock(page_content="a", metadata={"CELEX_ID": "A_source"}),
            Mock(page_content="b", metadata={"CELEX_ID": "B_source"}),
            Mock(page_content="c", metadata={"CELEX_ID": "C_source"}),
        ]

        self.assertEqual(p.postprocess(objects), "Document A_source:\n```\na\n```\nDocument B_source:\n```\nb\n```\nDocument C_source:\n```\nc\n```")

    def test_append_all_citation_postprocessor_with_delimiter(self) -> None:
        """Test if the AppendAllCitationPostprocessor works correctly with a delimiter."""
        p = AppendAllCitationPostprocessor(delimiter=" ")
        objects = [
            Mock(page_content="a", metadata={"CELEX_ID": "A_source"}),
            Mock(page_content="b", metadata={"CELEX_ID": "B_source"}),
            Mock(page_content="c", metadata={"CELEX_ID": "C_source"}),
        ]
        self.assertEqual(p.postprocess(objects), "Document A_source:\n```\na\n``` Document B_source:\n```\nb\n``` Document C_source:\n```\nc\n```")

    def test_append_all_citation_postprocessor_with_block_quotes(self) -> None:
        """Test if the AppendAllCitationPostprocessor works correctly with block quotes."""
        p = AppendAllCitationPostprocessor(block_quotes='"')
        objects = [
            Mock(page_content="a", metadata={"CELEX_ID": "A_source"}),
            Mock(page_content="b", metadata={"CELEX_ID": "B_source"}),
            Mock(page_content="c", metadata={"CELEX_ID": "C_source"}),
        ]
        self.assertEqual(p.postprocess(objects), "Document A_source:\n\"\na\n\"\nDocument B_source:\n\"\nb\n\"\nDocument C_source:\n\"\nc\n\"")
