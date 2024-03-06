<h1 align="center">
    <img style="width: 150px" src="elqm_icon.png" alt="Illustration icon: A modern light bulb design, with its filament shaped as a balance scale representing law. Encapsulating the bulb is a speech bubble, with a question mark and an answer tick, symbolizing the Q&A aspect.">
</h1>


<h1 align="center" style="margin-top: 0px;">ELQM: Energy-Law Query-Master</h1>
<h2 align="center" style="margin-top: 0px;">Natural Language Processing with Transformers</h2>

<div align="center">

[![pytest](https://github.com/psaegert/elqm-INLPT-WS2023/actions/workflows/pytest.yml/badge.svg)](https://github.com/psaegert/elqm-INLPT-WS2023/actions/workflows/pytest.yml)
[![quality checks](https://github.com/psaegert/elqm-INLPT-WS2023/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/psaegert/elqm-INLPT-WS2023/actions/workflows/pre-commit.yml)

</div>


# Introduction
We develop ELQM, a RAG-based question answering system for Eurpopean energy law acquired from [EUR-Lex](https://eur-lex.europa.eu/search.html?name=browse-by%3Alegislation-in-force&type=named&displayProfile=allRelAllConsDocProfile&qid=1696858573178&CC_1_CODED=12). ELQM comprises a full end-to-end pipeline, including data scraping, preprocessing, splitting, vectorization, storage, retrieval, and answer generation with chat-based LLMs. Our work also focuses on usability, providing three access points and linking to source documents for transparency.
# Requirements

### Hardware
- 16 GB RAM
- 12 GB VRAM
    - by default CUDA is used
- 25 GB storage space
    - 6 GB cache for all configurations
    - 7 GB environment
    - ~5 GB for *each* `llama2` and `mistral` model

## Software
- Python 3.10
- `pip` >= [24.0](https://github.com/google/sentencepiece/issues/378)
- [Ollama](https://ollama.ai/download)
- Ubuntu >= 22.04 (optional, for `GPT4AllEmbeddings` which requires glibc)
- For SparkNLP: Java OpenJDK or similar (see https://pypi.org/project/spark-nlp/)

# Getting Started
### 1. Clone the repository

```sh
git clone https://github.com/psaegert/elqm-INLPT-WS2023
cd elqm-INLPT-WS2023
```

### 2. Install the package

Optional: Create a virtual environment:

**conda:**

```sh
conda create -n elqm python=3.10 [ipykernel]
conda activate elqm
```

Optional: Install ipykernel to use the environment in Jupyter Notebook

**venv:**

```bash
python3 -m venv elqmVenv
source elqmVenv/bin/activate
```

Then, install the package via

```sh
pip install --upgrade pip
pip install -e .
```

### 3. Scrape the data
Scrape the EUR-Lex data with

```sh
elqm scrape-data
```

Alternatively, you can download the scraped data from our [Huggingface dataset](https://huggingface.co/datasets/ELQM/elqm-raw) and move its contents into `/data`

### 4. Install Ollama models

1. Run the Ollama backend via

```sh
ollama serve
```

2. Pull the desired Ollama model, e.g. `mistral`

```sh
ollama pull mistral
```

To generate the oracle dataset, we use the `llama2` model:

```sh
ollama pull llama2
```

# Usage

**Gradio Frontend**
```sh
elqm gui -c configs/prompts/256_5_5_nlc_bge_fn_mistral_h2.yaml
```

**CLI**
```sh
elqm run -c configs/prompts/256_5_5_nlc_bge_fn_mistral_h2.yaml
```

**Python API**
```python
from dynaconf import Dynaconf
import os

from elqm import ELQMPipeline
from elqm.utils import get_dir

config = Dynaconf(settings_files=os.path.join(get_dir("configs", "prompts"), "256_5_5_nlc_bge_fn_mistral_h2.yaml"))
elqm = ELQMPipeline(config)

print(elqm.answer("Which CIE LUV does a model supporting greater than 99 % of the sRGB colour space translate to?"))
```


# Development

### Setup
To set up the development environment, run the following commands:

```sh
pip install -e .[dev]
pre-commit install
```

### Tests

To run the tests locally, run the following commands:

```sh
ollama serve
pytest tests --cov src
```

# Citation
If you use ELQM: Energy-Law Query-Master for your research, please cite it using the following

```bibtex
@software{elqm_2024,
    author = {Daniel Knorr and Paul Saegert and Nikita Tatsch},
    title = {ELQM: Energy-Law Query-Master},
    month = mar,
    year = 2024,
    publisher = {GitHub},
    version = {1.0.0},
    url = {https://github.com/psaegert/elqm}
}
```
