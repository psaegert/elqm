import io
import sys
import unittest

from elqm.factories.preprocessing import PreprocessorFactory
from elqm.preprocessing import EnrichTextWithMetadataPreprocessor


class TestMetaDataEnricher(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_enricher_existence(self) -> None:
        preprocessor = PreprocessorFactory.get_preprocessor(
            preprocessor_name="enrich_with_metadata_preprocessor",
            metadata_keys=["author", "date"],
            load_key="text",
            save_key="enriched_text"
        )

        assert isinstance(preprocessor, EnrichTextWithMetadataPreprocessor)

    def test_enricher_functionality(self) -> None:
        preprocessor = EnrichTextWithMetadataPreprocessor(
            metadata_keys=["author", "date"],
            load_key="text",
            save_key="enriched_text"
        )

        document_dict = {
            "1.json": {
                "text": "This is a test document.",
                "author": "Nikita Tatsch",
                "date": "11.12.2000"
            },
            "2.json": {
                "text": "This is another test document.",
                "author": "John Doe",
                "date": "2021-01-01"
            }
        }

        expected_output = {
            "1.json": {
                "text": "This is a test document.",
                "author": "Nikita Tatsch",
                "date": "11.12.2000",
                "enriched_text": "Document content:\nThis is a test document.\nauthor:Nikita Tatsch\ndate:11.12.2000"
            },
            "2.json": {
                "text": "This is another test document.",
                "author": "John Doe",
                "date": "2021-01-01",
                "enriched_text": "Document content:\nThis is another test document.\nauthor:John Doe\ndate:2021-01-01"
            }
        }

        output = preprocessor.preprocess(document_dict)
        print(output)
        self.assertEqual(output, expected_output)

    def test_enricher_missing_metadata(self) -> None:
        preprocessor = EnrichTextWithMetadataPreprocessor(
            metadata_keys=["author", "date"],
            load_key="text",
            save_key="enriched_text"
        )

        document_dict = {
            "1.json": {
                "text": "This is a test document.",
                "author": "Nikita Tatsch"
            }
        }

        expected_output = {
            "1.json": {
                "text": "This is a test document.",
                "author": "Nikita Tatsch",
                "enriched_text": "Document content:\nThis is a test document.\nauthor:Nikita Tatsch"
            }
        }

        # Redirect stdout to capture the print statement
        captured_output = io.StringIO()
        sys.stdout = captured_output

        output = preprocessor.preprocess(document_dict)
        self.assertEqual(output, expected_output)

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Get the captured print statement
        print_statement = captured_output.getvalue().strip()
        self.assertEqual(print_statement, "Warning: Metadata key 'date' not found in document '1.json'. Will be skipped.")

    def test_enricher_deeper_key(self) -> None:
        preprocessor = EnrichTextWithMetadataPreprocessor(
            metadata_keys=["author.name", "author.age"],
            load_key="text",
            save_key="enriched_text"
        )

        document_dict = {
            "1.json": {
                "text": "This is a test document.",
                "author": {
                    "name": "Nikita Tatsch",
                    "age": 20
                }
            }
        }

        expected_output = {
            "1.json": {
                "text": "This is a test document.",
                "author": {
                    "name": "Nikita Tatsch",
                    "age": 20
                },
                "enriched_text": "Document content:\nThis is a test document.\nname:Nikita Tatsch\nage:20"
            }
        }

        output = preprocessor.preprocess(document_dict)
        self.assertEqual(output, expected_output)

    def test_data_enricher_list_handling(self) -> None:
        preprocessor = EnrichTextWithMetadataPreprocessor(
            metadata_keys=["author.name", "author.age"],
            load_key="text",
            save_key="enriched_text"
        )

        document_dict = {
            "1.json": {
                "text": "This is a test document.",
                "author": {
                    "name": ["Nikita Tatsch", "John Doe"],
                    "age": [20, 30]
                }
            }
        }

        expected_output = {
            "1.json": {
                "text": "This is a test document.",
                "author": {
                    "name": ["Nikita Tatsch", "John Doe"],
                    "age": [20, 30]
                },
                "enriched_text": "Document content:\nThis is a test document.\nname:Nikita Tatsch,John Doe\nage:20,30"
            }
        }

        output = preprocessor.preprocess(document_dict)
        print(output)
        self.assertEqual(output, expected_output)
