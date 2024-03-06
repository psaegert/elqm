import unittest
from unittest.mock import Mock

from elqm.postprocessing import AppendAllPostprocessor


class TestAppendAllPostprocessor(unittest.TestCase):

    def test_append_all_postprocessor(self) -> None:
        """Test if the AppendAllPostprocessor works correctly."""
        p = AppendAllPostprocessor()
        objects = [Mock(page_content="a"), Mock(page_content="b"), Mock(page_content="c")]
        self.assertEqual(p.postprocess(objects), "a\nb\nc")

    def test_append_all_postprocessor_with_delimiter(self) -> None:
        """Test if the AppendAllPostprocessor works correctly with a delimiter."""
        p = AppendAllPostprocessor(delimiter=" ")
        objects = [Mock(page_content="a"), Mock(page_content="b"), Mock(page_content="c")]
        self.assertEqual(p.postprocess(objects), "a b c")
