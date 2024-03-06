import json
import os
import time
import warnings

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from elqm.utils import get_dir


def extract_total_documents(eur_lex_query_url: str) -> int:
    """
    Extract the total number of documents from the query page

    Parameters
    ----------
    eur_lex_query_url : str
        The url of the query page

    Returns
    -------
    int
        The total number of documents
    """
    # FIXME: Unreliable, sometimes returns 402, although most of the time it is 522?
    response = requests.get(eur_lex_query_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    total_pages_element = soup.find('strong', string=True, recursive=True)
    total_pages = int(total_pages_element.find_next_sibling('strong').find_next_sibling('strong').string)
    return total_pages


def extract_links_from_results(url: str, base_url: str = "https://eur-lex.europa.eu") -> list[str] | None:
    """
    Extract the links to the documents from the query page

    Parameters
    ----------
    url : str
        The url of the query page
    base_url : str
        The base url of the website

    Returns
    -------
    list[str]
        A list of links to the documents
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Check if page number exceeds the maximum
    warning = soup.find('div', {'class': 'alert alert-warning'})
    if warning and 'maximum number of pages' in warning.text:
        return None

    # Find and save the links to DataFrame
    new_links = [f"{base_url}/{[link['href']][0][2:]}" for div in soup.find_all('div', {'class': 'SearchResult'}) for link in div.find_all('a', {'class': 'title'})]

    if len(new_links) == 0:
        return None

    return new_links


def get_document_info_page(url: str) -> str:
    """
    Get the redirected url from a given url

    Parameters
    ----------
    url : str
        The url to the document

    Returns
    -------
    str
        The redirected url
    """

    # Example: https://eur-lex.europa.eu/legal-content/AUTO/?uri=CELEX:31983H0230&qid=1698155004044&rid=460 -> https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:31983H0230

    url = url.replace('AUTO', 'EN/ALL')
    if url.find('&') == -1:
        return url
    return url[:url.find('&')]


def get_document_html_page(url: str) -> str:
    """
    Get the redirected url from a given url

    Parameters
    ----------
    url : str
        The url to the document

    Returns
    -------
    str
        The redirected url
    """

    # Example: https://eur-lex.europa.eu/legal-content/AUTO/?uri=CELEX:31983H0230&qid=1698155004044&rid=460 -> https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:31983H0230

    url = url.replace('AUTO', 'EN/TXT/HTML')
    if url.find('&') == -1:
        return url
    return url[:url.find('&')]


def get_document_CELEX_id(info_page_url: str) -> str:
    """
    Get the CELEX id from the info page url

    Parameters
    ----------
    info_page_url : str
        The url to the info page

    Returns
    -------
    str
        The CELEX id
    """

    # Extract 32023L1791 from https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32023L1791

    return info_page_url.split(':')[-1]


def scrape_links(energy_query_url: str = "https://eur-lex.europa.eu/search.html?name=browse-by%3Alegislation-in-force&type=named&qid=1698155004044&CC_1_CODED=12", n_retries: int = 5, interval: float = 0.5) -> None:
    """
    Scrapes the links to the documents from the query page and saves them to a csv file

    Parameters
    ----------
    energy_query_url : str
        The url of the query page
    n_retries : int
        The number of retries if the request fails
    """

    # Determine the total number of documents for validation
    n_documents = extract_total_documents(energy_query_url)

    if n_documents <= 402:
        # Something is wrong, try again
        for i in range(n_retries):
            print(f"Trying again, attempt {i+1}")
            n_documents = extract_total_documents(energy_query_url)
            if n_documents > 402:
                break

    print(f'Total number of documents matching "Energy" query: {n_documents}')

    # Scrape the links
    links = []
    pbar = tqdm(total=n_documents, desc="Scraping results for links")

    # Loop through the pages
    page_number = 1
    while True:
        url = f"{energy_query_url}&page={page_number}"
        pbar.set_postfix_str(f"{url}, attempt 1")

        new_links = extract_links_from_results(url)
        retry = 0
        while not new_links and retry <= n_retries:
            pbar.set_postfix_str(f"{url}, attempt {retry+2}")
            new_links = extract_links_from_results(url)
            retry += 1

        if not new_links:
            break

        links.extend(new_links)
        page_number += 1
        pbar.update(len(new_links))
        time.sleep(interval)

    pbar.close()

    df = pd.DataFrame(links, columns=['link'])
    if len(df) != n_documents:
        print(f"WARNING: Number of documents ({n_documents}) does not match the number of links ({len(df)})")

    df['info_page_url'] = df['link'].apply(get_document_info_page)
    df['html_page_url'] = df['link'].apply(get_document_html_page)
    df['CELEX_id'] = df['info_page_url'].apply(get_document_CELEX_id)

    # Drop duplicates (some documents are listed multiple times apparently)
    df = df.drop_duplicates(subset=['info_page_url'])
    df.to_csv(os.path.join(get_dir("data", create=True), 'eur_lex_links.csv'), index=False)


def get_document_content_html(html_page_url: str) -> str | None:
    """
    Get the html content of the document

    Parameters
    ----------
    html_page_url : str
        The url to the html page

    Returns
    -------
    str
        The html content of the document
    """
    response = requests.get(html_page_url)

    if response.status_code != 200:
        print('Error: status code {}'.format(response.status_code))
        return None

    return response.text


def get_dates_metadata(html: str) -> dict[str, str]:
    """
    Parse and extract the dates from the html

    Parameters
    ----------
    html : str
        The html content of the document

    Returns
    -------
    dict[str, str]
        A dictionary containing the dates
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Find the div tag with id 'PPMisc_Contents'
    dates_tag = soup.find('div', {'id': 'PPDates_Contents'})

    if dates_tag is None:
        return {}

    # Find the dl tag with class 'NMetadata'
    metadata_tag = dates_tag.find('dl', {'class': 'NMetadata'})

    # Initialize an empty dictionary to store dates
    dates_dict: dict[str, str] = {}
    key_counts: dict[str, int] = {}

    # Iterate over the dt and dd tags to scrape metadata
    for dt, dd in zip(metadata_tag.find_all('dt'), metadata_tag.find_all('dd')):
        key = dt.get_text(strip=True).replace(":", "")

        # If the key is already present, append a number to it
        if key in key_counts:
            key_counts[key] += 1
            key = f'{key} {key_counts[key]}'
        else:
            key_counts[key] = 0

        value = dd.get_text(strip=True)
        # Discard everything after the ; character if it is present
        if ';' in value:
            value = value.split(';')[0]
        dates_dict[key] = value

    return dates_dict


def get_miscellaneous_information_metadata(html: str) -> dict[str, str]:
    """
    Parse and extract the miscellaneous information from the html

    Parameters
    ----------
    html : str
        The html content of the document

    Returns
    -------
    dict[str, str]
        A dictionary containing the miscellaneous information
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Find the div tag with id 'PPMisc_Contents'
    misc_tag = soup.find('div', {'id': 'PPMisc_Contents'})

    if misc_tag is None:
        return {}

    # Find the dl tag with class 'NMetadata' within the div
    metadata_tag = misc_tag.find('dl', {'class': 'NMetadata'})

    # Initialize an empty dictionary to store metadata
    metadata_dict: dict[str, str] = {}
    key_counts: dict[str, int] = {}

    # Iterate over the dt and dd tags to scrape metadata
    for dt, dd in zip(metadata_tag.find_all('dt'), metadata_tag.find_all('dd')):
        key = dt.get_text(strip=True).replace(":", "")

        # If the key is already present, append a number to it
        if key in key_counts:
            key_counts[key] += 1
            key = f'{key} {key_counts[key]}'
        else:
            key_counts[key] = 0

        spans = dd.find_all('span', lang='en')
        if spans:
            value = ', '.join(span.get_text(strip=True) for span in spans)
        else:
            value = dd.get_text(strip=True)

        metadata_dict[key] = value

    return metadata_dict


def get_classifications_metadata(html: str) -> dict:
    """
    Parse and extract the classifications from the html

    Parameters
    ----------
    html : str
        The html content of the document

    Returns
    -------
    dict
        A dictionary containing the classifications
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Find the div tag with id 'PPClass_Contents'
    class_tag = soup.find('div', {'id': 'PPClass_Contents'})

    if class_tag is None:
        return {}

    # Find the dl tag with class 'NMetadata' within the div
    metadata_tag = class_tag.find('dl', {'class': 'NMetadata'})

    # Initialize an empty dictionary to store classifications
    classification_dict = {}
    code_dict = {}

    # Iterate over the dt and dd tags to scrape metadata
    for dt, dd in zip(metadata_tag.find_all('dt'), metadata_tag.find_all('dd')):
        major_key = dt.get_text(strip=True).replace(":", "")

        # For minor keys under each major key
        minor_keys = []
        for li in dd.find_all('li'):
            span = li.find('span', lang='en')
            if span:
                minor_keys.append(span.get_text(strip=True))

        # For extracting directory code and levels
        if major_key == "Directory code":
            code = dd.find('li').get_text().split('\n')[0].strip()
            code_dict["code"] = code

            levels = dd.find_all('span', lang='en')
            for idx, level in enumerate(levels):
                code_dict[f"level {idx + 1}"] = level.get_text(strip=True)

            classification_dict[major_key] = code_dict
        else:
            classification_dict[major_key] = minor_keys  # type: ignore

    return classification_dict


def get_metadata(info_page_url: str) -> dict[str, dict]:
    """
    Get the metadata from the info page

    Parameters
    ----------
    info_page_url : str
        The url to the info page

    Returns
    -------
    dict[str, dict]
        A dictionary containing the metadata
    """
    response = requests.get(info_page_url)

    if response.status_code != 200:
        print('Error: status code {}'.format(response.status_code))
        raise Exception(f"Error: status code {response.status_code}")

    metadata = {
        'Dates': get_dates_metadata(response.text),
        'Misc': get_miscellaneous_information_metadata(response.text),
        'Classification': get_classifications_metadata(response.text)
    }

    return metadata


def get_document(info_page_url: str, html_page_url: str) -> dict[str, dict]:
    """
    Get the metadata and content from the document

    Parameters
    ----------
    info_page_url : str
        The url to the info page
    html_page_url : str
        The url to the html page

    Returns
    -------
    dict[str, dict]
        A dictionary containing the metadata and content
    """
    metadata = get_metadata(info_page_url)
    metadata['html'] = get_document_content_html(html_page_url)  # type: ignore

    return metadata


def scrape_data(data_dir: str | None = None, interval: float = 0.5, max_passes: int = 5) -> None:
    """
    Scrape the data from the website into a json file for each document

    Parameters
    ----------
    data_dir : str
        The path to the data directory
    interval : float
        The time interval between each request
    max_passes : int
        The maximum number of times to iterate over the links and try to re-download incomplete documents
    """

    # Check if the eur_lex_links.csv file exists
    if not os.path.exists(os.path.join(get_dir("data"), 'eur_lex_links.csv')):
        print("eur_lex_links.csv file not found, scraping links first")
        scrape_links(interval=interval)

    df = pd.read_csv(os.path.join(get_dir("data"), 'eur_lex_links.csv'))

    if data_dir is None:
        data_dir = get_dir("data", "eur_lex_data", create=True)

    for i in range(max_passes):
        incomplete_files_count = 0
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading documents"):
            # Skip if the file already exists
            if os.path.exists(os.path.join(data_dir, f'{row["CELEX_id"]}.json')):
                # Check if all metadata is present
                with open(os.path.join(data_dir, f'{row["CELEX_id"]}.json'), 'r') as f:
                    locally_available_document = json.load(f)

                if len(locally_available_document['Dates']) == 0 and len(locally_available_document['Misc']) == 0 and len(locally_available_document['Classification']) == 0:
                    # Try to download the document again
                    print(f"Warning: metadata is empty for {row['CELEX_id']}. Repeating download...")
                else:
                    continue

            document = get_document(row['info_page_url'], row['html_page_url'])

            # Check if the document contains the html content
            if document['html'] is None or document['html'] == '':
                print(f"Error: html content not found for {row['CELEX_id']}. Skipping...")
                incomplete_files_count += 1
                continue

            # If the metadata is empty, warn the user
            if len(document['Dates']) == 0 and len(document['Misc']) == 0 and len(document['Classification']) == 0:
                warnings.warn(f"Warning: metadata is empty for {row['CELEX_id']}")
                incomplete_files_count += 1

            with open(os.path.join(data_dir, f'{row["CELEX_id"]}.json'), 'w') as f:
                json.dump(document, f)
            time.sleep(interval)

        if incomplete_files_count == 0:
            break

        print(f"Repeating download for {incomplete_files_count} incomplete documents")

    if incomplete_files_count > 0:
        warnings.warn(f"Warning: {incomplete_files_count} documents are still incomplete after {max_passes} passes. Please try again later.")
