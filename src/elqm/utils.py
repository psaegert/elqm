import json
import os
import shutil
import subprocess
from typing import Any

from dynaconf import Dynaconf
from langchain_core.documents import Document


def get_dir(*args: str, create: bool = False) -> str:
    """
    Get the path to the data directory.

    Parameters
    ----------
    args : str
        The path to the data directory.
    create : bool, optional
        Whether to create the directory if it does not exist, by default False.

    Returns
    -------
    str
        The path to the data directory.
    """
    if any([not isinstance(arg, str) for arg in args]):
        raise TypeError("All arguments must be strings.")

    if create:
        os.makedirs(os.path.join(os.path.dirname(__file__), '..', '..', *args), exist_ok=True)

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', *args))


def get_raw_data(data_dir: str | None = None) -> dict[str, dict]:
    """
    Read all json files in the data directory and return them as a dictionary

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the json files

    Returns
    -------
    dict
        Dictionary containing the data from the json files
    """
    # Apply the default path if none is provided
    if data_dir is None:
        data_dir = get_dir("data", "eur_lex_data")

    # Read all json files in the data directory
    raw_data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename)) as f:
                raw_data[filename.replace('.json', '')] = json.load(f)

    return raw_data


def is_config_cached(config: Dynaconf) -> bool:
    """
    Check if a cache file exists for the given config.

    Parameters
    ----------
    cache_key : Dynaconf
        The config object that specifies the retrievers to check for.

    Returns
    -------
    bool
        True if the cache file exists, False otherwise.
    """
    if config.retriever == "ensemble":
        for retriever in config.retriever_args["retrievers"]:
            if not cache_exists(config.index_name, retriever['retriever']):
                return False
    else:
        if not cache_exists(config.index_name, config.retriever):
            return False

    return True


def get_cache_filename(retriever: str) -> str:
    """
    Get the cache filename for the given retriever.

    Parameters
    ----------
    retriever : str
        The name of the retriever.

    Returns
    -------
    str
        The cache filename.
    """
    match retriever:
        case "FAISS":
            return "index.faiss"
        case "chroma":
            return "chroma.sqlite3"
        case "BM25":
            return "BM25.pickle"
        case _:
            raise ValueError(f"Retriever {retriever} is not supported.")


def cache_exists(index_name: str, retriever: str | None) -> bool:
    """
    Check if a cache file exists for the given index name and retriever.

    Parameters
    ----------
    index_name : str
        The name of the index.
    retriever : str | None
        The name of the retriever.

    Returns
    -------
    bool
        True if the cache file exists, False otherwise.
    """
    if retriever is None:
        cache_path = get_dir("cache", index_name)
    else:
        cache_path = os.path.join(get_dir("cache", index_name, retriever), get_cache_filename(retriever))

    return os.path.exists(cache_path)


def clear_cache(index_name: str) -> None:
    """
    Clear the cache for the given index name.

    Parameters
    ----------
    index_name : str
        The name of the index.
    """
    try:
        shutil.rmtree(get_dir("cache", index_name))
    except FileNotFoundError:
        pass


def is_model_installed(model_name: str) -> bool:
    """
    Check if a model is installed.

    Parameters
    ----------
    model_name : str
        The name of the model, e.g. 'llama2' or 'llama2:latest'.

    Returns
    -------
    bool
        True if the model is installed, False otherwise.

    Notes
    -----
    `ollama list` returns a list of all installed models, e.g.

    ```
    $ ollama list
    NAME                                    ID              SIZE    MODIFIED
    dolphin-mixtral:latest                  4b33b01bf336    26 GB   5 weeks ago
    dolphin-mixtral-uncensored:latest       9cb9a0f0ab16    26 GB   5 weeks ago
    dolphin2.2-mistral:latest               5e3b248c93d7    4.1 GB  2 months ago
    llama2:latest                           fe938a131f40    3.8 GB  7 weeks ago
    mistral:instruct                        d364aa8d131e    4.1 GB  2 months ago
    orca2:latest                            ea98cc422de3    3.8 GB  2 months ago
    phi:latest                              c6afdcfc5564    1.6 GB  4 weeks ago
    ```
    """
    # First, check if the ollama service is running
    if not is_ollama_serve_running():
        raise RuntimeError("Ollama serve is not running. Start it with `ollama serve`.")

    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    except FileNotFoundError:  # Ollama is not installed
        return False

    # Parse the output of ollama list
    for line in result.stdout.split('\n'):
        if ':' in model_name:
            if line.split(r'\s')[0] == model_name:
                return True
        else:
            if line.startswith(model_name):
                return True

    return False


def is_ollama_serve_running() -> bool:
    """
    Check if ollama serve is running.

    Returns
    -------
    bool
        True if ollama serve is running, False otherwise.

    Notes
    -----
    `ps aux` returns a list of all running processes, e.g.

    ```
    $ ps aux | grep "ollama"
    psaegert 65742  6.5  0.8 2156920 411608 pts/5  Sl+  18:19   0:00 ollama serve
    psaegert 65757  0.0  0.0   7004  2128 pts/6    S+   18:19   0:00 grep --color=auto ollama
    ```
    """
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)

    # Parse the output of ps aux  # HACK: Does not account for false positives
    for line in result.stdout.split('\n'):
        if 'ollama serve' in line:
            return True

    return False


def create_citation_link(celex_id: str) -> str:
    """
    Create a link to the EUR-Lex website for the given CELEX ID.

    Parameters
    ----------
    celex_id : str
        The CELEX ID.

    Returns
    -------
    str
        The link to the EUR-Lex website for the given CELEX ID, e.g https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:31983H0230
    """
    return f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex_id}"


def extract_metadata(record: dict, metadata: dict, mapping: dict | None = None) -> dict:
    """
    Extract metadata from a record

    Parameters
    ----------
    record : dict
        The record to extract metadata from.
    metadata : dict | None
        The metadata to extract from the record.
    config : dict
        Configuration options for the metadata extraction.

    Returns
    -------
    dict
        The extracted metadata.

    Example
    -------
    {
        "Dates": {
            "Date of document": "01/12/1975",
            "Date of effect": "01/01/1976",
            "Date of signature": "01/12/1975",
            "Date of end of validity": "No end date"
        },
        "Misc": {
            "Author": "European Atomic Energy Community, International Atomic Energy Agency",
            "Form": "International agreement",
            "Internal comment": "Bilateral agreement",
            "Depositary": "United Nations Organization - Secretary-General",
            "Additional information": "Validity : notice of termination of 6 Months",
            "Authentic language": "English, French"
        },
        "Classification": {
            "EUROVOC descriptor": [
                "International Atomic Energy Agency",
                "cooperation agreement",
                "nuclear energy"
            ],
            "Subject matter": [
                "Dissemination of information",
                "External relations",
                "Nuclear common market"
            ],
            "Directory code": {
                "code": "11.30.40.00",
                "level 1": "External relations",
                "level 2": "Multilateral relations",
                "level 3": "Cooperation with international and non-governmental organisations",
                "level 4": "Energy",
                "level 5": "Nuclear energy"
            }
        },
        "html": "..."
    }
    """
    if mapping is None:
        return metadata

    for key, value in mapping.items():
        # The value has an attribute json_key: ["Dates", "Date of document"]
        # Extract the value from the record and assign it to the metadata

        metadata[key] = get_nested_value(record, value["json_key"])

        if value["type"] == "date":
            metadata[key] = format_date_yyyy_mm_dd(metadata[key])

    return metadata


def format_date_yyyy_mm_dd(date: str | None) -> str:
    """
    Format a date string from dd/mm/yyyy to yyyy-mm-dd

    Parameters
    ----------
    date : str | None
        The date string to format.

    Returns
    -------
    str
        The formatted date string.
    """
    if date is None:
        return ""
    return "-".join(date.split("/")[::-1])


def get_nested_value(data: dict, keys: list[str], default: Any = "") -> Any:
    """
    Get a nested value from a dictionary

    Parameters
    ----------
    data : dict
        The dictionary to get the value from.
    keys : list[str]
        The keys to traverse the dictionary.
    default : Any, optional
        The default value to return if the value is not found, by default None.

    Returns
    -------
    Any
        The value at the nested keys or the default value if the value is not found.
    """
    if len(keys) == 1:
        return data.get(keys[0], default)
    elif keys[0] in data:
        return get_nested_value(data[keys[0]], keys[1:], default)
    else:
        return default


def deduplicate_retrieved_documents(documents: list[Document]) -> list[Document]:
    """
    Remove duplicates from a list of documents while preserving the order.

    Parameters
    ----------
    documents : list[Document]
        The list of documents to remove duplicates from.

    Returns
    -------
    list[Document]
        The list of documents without duplicates.
    """
    # Sets do not work here since Document is not hashable
    unique_documents = []

    for document in documents:
        if document not in unique_documents:
            unique_documents.append(document)

    return unique_documents
