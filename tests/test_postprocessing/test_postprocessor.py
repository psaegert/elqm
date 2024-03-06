import unittest

from elqm.postprocessing.postprocessor import Postprocessor


class TestPostprocessor(unittest.TestCase):

    def test_postprocessor_not_implemented(self) -> None:
        """Test if the abstract class Postprocessor raises NotImplementedError."""
        p = Postprocessor()
        with self.assertRaises(NotImplementedError):
            p.postprocess([])
