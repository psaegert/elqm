from typing import Any

from elqm.factories.splitters import SplitterFactory
from elqm.preprocessing.preprocessors import DictSplitter, EnrichTextWithMetadataPreprocessor, HTMLSplitter, Preprocessor, RemoveHTMLTagsPreprocessor, TransformFootnotesPreprocessor


class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(preprocessor_name: str, *args: Any, **kwargs: Any) -> Preprocessor:
        """
        Factory method to get a preprocessor

        Parameters
        ----------
        preprocessor_name : str
            Name of the preprocessor to use. Determines which preprocessor class to instantiate.
        *args : Any
            Positional arguments to pass to the preprocessor class.
        **kwargs : Any
            Keyword arguments to pass to the preprocessor class.

        Returns
        -------
        Preprocessor
            An instance of the preprocessor class specified by preprocessor_name.
        """
        match preprocessor_name:
            case "transform_footnotes_preprocessor":
                return TransformFootnotesPreprocessor(*args, **kwargs)
            case "remove_html_tags_preprocessor":
                return RemoveHTMLTagsPreprocessor(*args, **kwargs)
            case "dict_splitter":
                return DictSplitter(
                    splitter=SplitterFactory().get_splitter(
                        splitter_name=kwargs['splitter'], **kwargs['splitter_kwargs']),
                    split_field=kwargs['split_field']
                )
            case "html_splitter":
                return HTMLSplitter(
                    headers_to_split_on=kwargs['headers_to_split_on'].items(),
                    split_field=kwargs['split_field']
                )
            case "enrich_with_metadata_preprocessor":
                return EnrichTextWithMetadataPreprocessor(
                    metadata_keys=kwargs['metadata_keys'],
                    load_key=kwargs['load_key'],
                    save_key=kwargs['save_key'])
            case _:
                raise NotImplementedError
