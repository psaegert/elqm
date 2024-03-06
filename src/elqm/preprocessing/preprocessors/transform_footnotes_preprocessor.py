import re

from bs4 import BeautifulSoup
from tqdm import tqdm

from elqm.preprocessing.preprocessors.preprocessor import Preprocessor


class TransformFootnotesPreprocessor(Preprocessor):
    """
    For transforming the footnotes in the documents.
    """

    def __init__(self) -> None:
        pass

    def __generate_id2text_dict(self, soup: BeautifulSoup) -> dict:
        """
        Build a dict with the help of the html fields, where:
            Key: The ID of a footnote (this is the content of the field "id" of a paragraph (<p>) with the class "oj-note" in the a_tag)
            Value: This is the text in that paragraph

        Args:
            soup: Represents the parsed documents as a whole
        Returns:
            dictionary: <Footnote-ID>:<Footnote-Content>
        """

        # Extract all <p> elements
        paragraphs = soup.find_all('p', class_='oj-note')

        id_text_dict = {}

        for paragraph in paragraphs:

            # Find the <a> tag within the <p> element
            a_tag = paragraph.find('a')

            # sometimes paragraphs with oj-note have no a_tag,
            # and only if it has an a_tag there is an ID we can use for Transformation
            if a_tag:

                # Extract the uri or text
                uri_or_text = paragraph.get_text(strip=True)

                # Clean string regarding certain characters
                uri_or_text = uri_or_text.replace(u'\xa0', u' ')

                # Clean string regarding "(<Number>)..." at the beginning of the string
                uri_or_text = re.sub(r'^\(\d+\)', '', uri_or_text)

                # Extract id from the a_tag
                id_value = a_tag.get('id')

                # Fill dict
                id_text_dict[id_value] = uri_or_text

        return id_text_dict

    def __replace_footnote_with_footnote_content(self, soup: BeautifulSoup, id_text_dict: dict) -> BeautifulSoup:
        """
        Replaces a footnote ("(<Number>)") with the footnote content in brackets

        Args:
            soup: Represents the parsed documents as a whole
            id_text_dict: <Footnote-ID>:<Footnote-Content> --> the output of the method "generate_id2text_dict"
        Returns:
            soup: Parsed documents, where the footnotes are replaced with their specific content in brackets
        """

        # The "id" in a oj-note is equal to the "href" of a footnote in the text, but the
        # "href" starts with an "#"
        # So extract all <a> elements with href starting with "#"
        a_tags = soup.find_all(
            'a', href=lambda href: href and href.startswith("#"))

        # Replace the <a> tags with values from id_text_dict and delete them afterwards
        for a_tag in a_tags:
            href_value = a_tag.get('href')

            # Remove the "#" prefix to find id (from oj-note) == href (without "#")
            id_value = href_value[1:] if href_value and href_value.startswith(
                "#") else None

            if id_value in id_text_dict:
                new_text = f"({id_text_dict[id_value]})"

                # the footnote text is inserted after the a_tag
                a_tag.insert_after(new_text)

                # now it is: <normal text> <a_tag> <(footnote content)>, so
                # we delete the a_tag now and the consequence is, that the footnote
                # content is automatically behind the normal text, like
                # <normal text> <(footnote content)>
                a_tag.decompose()

        return soup

    def __remove_footnotes(self, soup: BeautifulSoup) -> BeautifulSoup:
        """
        Removes the footnotes (Paragraphs with class 'oj-note')

        Args:
            soup: Parsed documents, where the footnotes are replaced with their specific content in brackets
        Returns:
            soup: Parsed documents, where the footnotes are removed
        """

        # Remove <p> tags with class 'oj-note' to delete all footnotes
        oj_note_paragraphs = soup.find_all('p', class_='oj-note')
        for oj_note_paragraph in oj_note_paragraphs:
            # Now all footnotes will be deleted
            oj_note_paragraph.decompose()

        return soup

    def preprocess(self, document_dict: dict[str, dict], verbose: bool = True) -> dict[str, dict]:
        """
        Reorders footnotes in the HTML docs: Replaces the footnotes in the
        text with the actual footnote content in brackets and then deletes
        all footnotes.

        Parameters
        ----------
        document_dict : dict[str, dict]
            The list of texts to preprocess.
        verbose : bool
            The verbosity of the preprocessing.

        Returns
        -------
        dict[str, dict]
            The list of docs with HTML with pre-processed footnotes.
        """

        preprocessed_document_dict = document_dict.copy()
        for id, document in tqdm(preprocessed_document_dict.items(), disable=not verbose, desc="Transform Footnotes in HTML Docs"):

            soup = BeautifulSoup(document['html'], 'html.parser')

            # let's start the footnote transformation, which is divided into 3 steps
            id_text_dict = self.__generate_id2text_dict(soup)
            soup = self.__replace_footnote_with_footnote_content(
                soup, id_text_dict)
            soup = self.__remove_footnotes(soup)

            # for removing html tags after the transformation of
            # the footnotes, we need to save "soup" as a string
            soup = str(soup)
            preprocessed_document_dict[id]['html'] = soup

        return preprocessed_document_dict
