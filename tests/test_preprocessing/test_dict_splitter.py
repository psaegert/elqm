import unittest
from unittest.mock import MagicMock

from elqm.preprocessing import DictSplitter


class TestDictSplitter(unittest.TestCase):

    def setUp(self) -> None:
        # Setup a mock TextSplitter with predetermined behavior
        self.mock_splitter = MagicMock()
        self.mock_splitter.split_text.side_effect = self.mock_split_function
        self.split_field = 'content'
        self.dict_splitter = DictSplitter(splitter=self.mock_splitter, split_field=self.split_field)

    def mock_split_function(self, text: str) -> list[str]:
        # Define how text should be split; simple example: split by sentence.
        return text.split('. ')

    def test_basic_functionality(self) -> None:
        document_dict = {
            'doc1.txt': {'content': 'First sentence. Second sentence.'}
        }
        expected_output = {
            'doc1_0.txt': {'content': 'First sentence', 'ID': 0},
            'doc1_1.txt': {'content': 'Second sentence.', 'ID': 1}
        }
        output = self.dict_splitter.preprocess(document_dict)
        self.assertEqual(output, expected_output)

    def test_multiple_documents(self) -> None:
        document_dict = {
            'doc1.txt': {'content': 'First doc, first sentence. First doc, second sentence.'},
            'doc2.txt': {'content': 'Second doc, first sentence. Second doc, second sentence.'}
        }
        expected_output = {
            'doc1_0.txt': {'content': 'First doc, first sentence', 'ID': 0},
            'doc1_1.txt': {'content': 'First doc, second sentence.', 'ID': 1},
            'doc2_0.txt': {'content': 'Second doc, first sentence', 'ID': 2},
            'doc2_1.txt': {'content': 'Second doc, second sentence.', 'ID': 3}
        }
        output = self.dict_splitter.preprocess(document_dict)
        self.assertEqual(output, expected_output)

    def test_empty_document(self) -> None:
        document_dict = {
            'empty.txt': {'content': ''}
        }
        output = self.dict_splitter.preprocess(document_dict)
        self.assertEqual(output, {'empty_0.txt': {'content': '', 'ID': 0}})

    def test_verbose_mode(self) -> None:
        # This test assumes verbose mode would print logs; since logging is not captured here, we'll just ensure no error
        document_dict = {
            'doc.txt': {'content': 'Sentence.'}
        }
        # No assertion for output; just ensuring it runs without error
        self.dict_splitter.preprocess(document_dict, verbose=True)

    def test_special_characters_and_formats(self) -> None:
        document_dict = {
            'special.txt': {'content': 'HTML content: <div>Some text.</div>'}
        }
        expected_output = {
            'special_0.txt': {'content': 'HTML content: <div>Some text.</div>', 'ID': 0}
        }
        output = self.dict_splitter.preprocess(document_dict)
        self.assertEqual(output, expected_output)

    def test_large_document_splitting(self) -> None:
        large_text = ' '.join(['Sentence.'] * 1000)  # Simulate a large document
        document_dict = {
            'large.txt': {'content': large_text}
        }
        expected_output_keys = [f'large_{i}.txt' for i in range(1000)]
        output = self.dict_splitter.preprocess(document_dict)
        self.assertEqual(list(output.keys()), expected_output_keys)

    def test_no_splitting_required(self) -> None:
        document_dict = {
            'short.txt': {'content': 'Short sentence'}
        }
        expected_output = {
            'short_0.txt': {'content': 'Short sentence', 'ID': 0}
        }
        output = self.dict_splitter.preprocess(document_dict)
        self.assertEqual(output, expected_output)
