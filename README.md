# Ragferee

Retrieval-augmented Q&A over the IFAB Laws of the Game. Ask a football rules question and get a plain-language explanation backed by the exact rule text, cited by Law, section and page. Built with Python, FastAPI and pgvector, with hybrid retrieval, an evaluation suite, and a clean object-oriented design.

## Status

Retrieval works end to end; answering is next.

| Area                                                       | Status  |
|------------------------------------------------------------|---------|
| Download the official PDF (checksum verified)              | Done    |
| Parse Laws 1–17 into chunks cited by Law, section and page | Done    |
| Local embeddings stored in Postgres with pgvector          | Done    |
| Hybrid search (vector + keyword, Reciprocal Rank Fusion)   | Done    |
| Retrieval evaluation (hit rate, MRR, mode comparison)      | Done    |
| Answers with citations from an LLM (`ragferee ask`)        | Planned |
| Web API (FastAPI)                                          | Planned |

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs Python 3.13 for you)
- Docker with Docker Compose
- make

## Setup

```bash
make install
cp .env.example .env    # then fill in POSTGRES_USER and POSTGRES_PASSWORD
make db-up
uv run ragferee check
```

`ragferee check` validates your settings and prints a summary without revealing secrets.

## Configuration

Settings are read from `.env` and environment variables (environment variables win).

| Variable            | Required                 | Default     | Description        |
|---------------------|--------------------------|-------------|--------------------|
| `POSTGRES_USER`     | yes                      |             | Database user      |
| `POSTGRES_PASSWORD` | yes                      |             | Database password  |
| `POSTGRES_DB`       | no                       | `ragferee`  | Database name      |
| `POSTGRES_HOST`     | no                       | `localhost` | Database host      |
| `POSTGRES_PORT`     | no                       | `5432`      | Database port      |
| `LLM_PROVIDER`      | no                       | `mock`      | `mock` or `openai` |
| `OPENAI_API_KEY`    | if `LLM_PROVIDER=openai` |             | OpenAI API key     |

The database connection URL is built from these values, so it is not configured separately.

## Usage

Build the search index once (the first run also downloads a 65 MB embedding model):

```bash
uv run ragferee download   # fetch the Laws of the Game PDF into data/raw/
uv run ragferee ingest     # split it into chunks in data/processed/chunks.json
uv run ragferee index      # embed the chunks and store them in Postgres
```

Then search it:

```bash
uv run ragferee search "What happens if a goalkeeper holds the ball too long?"
```

```text
1. Law 12, 3. Corner kick, p. 119  (0.0323)
   A corner kick is awarded if a goalkeeper, inside their penalty area, controls the ball with their …
2. Law 12, 1. Direct free kick, p. 116  (0.0320)
   …
```

| Command                                      | What it does                                        |
|----------------------------------------------|-----------------------------------------------------|
| `check`                                      | Validate settings without revealing secrets         |
| `download [--force]`                         | Download the PDF and verify its checksum            |
| `ingest`                                     | Parse the PDF into chunks                           |
| `index`                                      | Embed the chunks and store them in Postgres         |
| `search "<question>" [--limit N]`            | Show the most relevant chunks with citations        |
| `eval [--limit N] [--mode MODE] [--compare]` | Measure retrieval quality on `eval/retrieval.jsonl` |

## How it works

1. **Ingest** – PyMuPDF reads pages 44–151 (Laws 1–17). Laws and sections are detected from font size and text,
   giving one chunk per section per page, so every result can be cited exactly.
2. **Embed** – each chunk is embedded locally with `BAAI/bge-small-en-v1.5` (fastembed, 384 dimensions); no API key needed.
3. **Store** – chunks and embeddings live in Postgres with pgvector; a generated full-text column supports keyword search.
4. **Search** – a question is matched by meaning (cosine distance) and by keywords, and the two rankings are merged
   with Reciprocal Rank Fusion.

## Retrieval quality

`uv run ragferee eval --compare` on 36 hand-checked questions covering all 17 Laws
(`eval/retrieval.jsonl`, many phrased in everyday words such as "penalty spot" or "shin pads"):

| Mode    | Hit@1 | Hit@5 | MRR  |
|---------|-------|-------|------|
| Hybrid  | 0.72  | 0.92  | 0.81 |
| Vector  | 0.64  | 0.97  | 0.77 |
| Keyword | 0.50  | 0.75  | 0.60 |

Hybrid search ranks the right section first most often. Vector search alone finds slightly more answers in the
top 5, so how keyword results are weighted is worth revisiting. Tables in the PDF, such as pitch dimensions,
extract poorly and are a known weak spot.

## Development

| Command                       | What it does                                   |
|-------------------------------|------------------------------------------------|
| `make install`                | Install dependencies                           |
| `make lint`                   | Ruff lint, Ruff format check and mypy (strict) |
| `make format`                 | Format and auto-fix lint issues                |
| `make test`                   | Run the test suite                             |
| `make hooks`                  | Run the pre-commit checks on all files         |
| `make db-up` / `make db-down` | Start / stop Postgres with pgvector            |

Pre-commit hooks run the same checks on every commit:

```bash
uv run pre-commit install
```

## Data

The IFAB Laws of the Game are not included in this repository; `ragferee download` fetches the official PDF from IFAB. Everything under `data/` (the PDF, parsed chunks and the embedding model) is ignored by git.

## Author

**Arabind Meher**

[LinkedIn](https://www.linkedin.com/in/arabind-meher/) | [GitHub](https://github.com/arabind-meher)
<br>
arabindmeher.99@gmail.com

## License

[MIT](LICENSE)
