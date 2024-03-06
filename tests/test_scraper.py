from bs4 import BeautifulSoup

from elqm.scraper import extract_links_from_results, extract_total_documents, get_classifications_metadata, get_dates_metadata, get_document, get_document_CELEX_id, get_document_content_html, get_document_html_page, get_document_info_page, get_metadata, get_miscellaneous_information_metadata

EUR_LEX_QUERY_URL = "https://eur-lex.europa.eu/search.html?name=browse-by%3Alegislation-in-force&type=named&qid=1698155004044&CC_1_CODED=12"


def test_get_document_info_page() -> None:
    # Example: https://eur-lex.europa.eu/legal-content/AUTO/?uri=CELEX:31983H0230&qid=1698155004044&rid=460 -> https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:31983H0230

    url = "https://eur-lex.europa.eu/legal-content/AUTO/?uri=CELEX:31983H0230&qid=1698155004044&rid=460"
    info_page = get_document_info_page(url)

    assert info_page == "https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:31983H0230"


def test_get_document_html_page() -> None:
    # Example: https://eur-lex.europa.eu/legal-content/AUTO/?uri=CELEX:31983H0230&qid=1698155004044&rid=460 -> https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:31983H0230

    url = "https://eur-lex.europa.eu/legal-content/AUTO/?uri=CELEX:31983H0230&qid=1698155004044&rid=460"
    html_page = get_document_html_page(url)

    assert html_page == "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:31983H0230"


def test_get_document_CELEX_id() -> None:
    # Extract 32023L1791 from https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32023L1791

    url = "https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32023L1791"

    assert get_document_CELEX_id(url) == "32023L1791"


def test_extract_total_documents() -> None:
    total_pages = extract_total_documents(EUR_LEX_QUERY_URL)

    assert total_pages > 0


def test_get_document_content_html() -> None:
    test_html_page_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32023L1791"

    html = get_document_content_html(test_html_page_url)

    assert html is not None
    # Check if the html can be parsed
    soup = BeautifulSoup(html, "html.parser")
    assert soup is not None


def test_extract_links_from_results() -> None:
    test_html_page_url = "https://eur-lex.europa.eu/search.html?name=browse-by%3Alegislation-in-force&type=named&qid=1698155004044&CC_1_CODED=12"

    links = extract_links_from_results(test_html_page_url)

    assert len(links) > 0
    # Remove url parameters
    assert get_document_html_page(links[0]) == "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32023L1791"


def test_get_dates_metadata() -> None:
    # Get some html
    test_html_page_url = "https://eur-lex.europa.eu/legal-content/AUTO/?uri=CELEX:32022D0604"
    test_info_page_url = get_document_info_page(test_html_page_url)
    html = get_document_content_html(test_info_page_url)
    assert html is not None

    # Extract dates
    dates = get_dates_metadata(html)

    assert isinstance(dates, dict)

    # FIXME: EUR-Lex sometimes returns incomplete HTML rendering this test unreliable

    # assert "Date of document" in dates
    # assert "Date of effect" in dates

    # assert dates["Date of document"] == "08/04/2022"
    # # Test if comments are stripped
    # assert dates["Date of effect"] == "13/04/2022"


def test_get_miscellaneous_information_metadata() -> None:
    # Get some html
    test_html_page_url = "https://eur-lex.europa.eu/legal-content/AUTO/?uri=CELEX:32010L0031"
    test_info_page_url = get_document_info_page(test_html_page_url)
    html = get_document_content_html(test_info_page_url)
    assert html is not None

    # Extract miscellaneous information
    misc_info = get_miscellaneous_information_metadata(html)

    assert isinstance(misc_info, dict)

    # FIXME: EUR-Lex sometimes returns incomplete HTML rendering this test unreliable

    # assert "Author" in misc_info
    # assert "Form" in misc_info
    # assert "Addressee" in misc_info
    # assert "Additional information" in misc_info

    # assert misc_info["Author"] == "European Parliament, Council of the European Union"
    # assert misc_info["Form"] == "Directive"


def test_get_classifications_metadata() -> None:
    # Get some html
    test_html_page_url = "https://eur-lex.europa.eu/legal-content/AUTO/?uri=CELEX:31983H0230"
    test_info_page_url = get_document_info_page(test_html_page_url)
    html = get_document_content_html(test_info_page_url)
    assert html is not None

    # Extract classifications
    classifications = get_classifications_metadata(html)

    assert isinstance(classifications, dict)

    # FIXME: EUR-Lex sometimes returns incomplete HTML rendering this test unreliable

    # assert "EUROVOC descriptor" in classifications
    # assert "Subject matter" in classifications
    # assert "Directory code" in classifications

    # assert type(classifications["EUROVOC descriptor"]) is list
    # assert type(classifications["Subject matter"]) is list
    # assert type(classifications["Directory code"]) is dict

    # assert "fixing of prices" in classifications["EUROVOC descriptor"]
    # assert "Energy" in classifications["Subject matter"]
    # assert classifications["Directory code"]["code"] == "12.50.30.00"
    # assert classifications["Directory code"]["level 1"] == "Energy"


def test_get_metadata() -> None:
    # Get some html
    test_html_page_url = "https://eur-lex.europa.eu/legal-content/AUTO/?uri=CELEX:31983H0230"
    test_info_page_url = get_document_info_page(test_html_page_url)
    # Extract metadata
    metadata = get_metadata(test_info_page_url)

    assert isinstance(metadata, dict)

    assert "Dates" in metadata
    assert "Misc" in metadata
    assert "Classification" in metadata


def test_get_document() -> None:
    # Get some html
    test_html_page_url = "https://eur-lex.europa.eu/legal-content/AUTO/?uri=CELEX:32023L1791"
    test_info_page_url = get_document_info_page(test_html_page_url)

    document = get_document(test_info_page_url, test_html_page_url)

    assert document is not None

    assert "Dates" in document
    assert "Misc" in document
    assert "Classification" in document
    assert "html" in document

    assert BeautifulSoup(document["html"], "html.parser") is not None
