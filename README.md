# Ragferee

Retrieval-augmented Q&A over the IFAB Laws of the Game. Ask a football rules question and get a plain-language explanation backed by the exact rule text, cited by Law, section and page. Built with Python, FastAPI and pgvector, with hybrid retrieval, an evaluation suite, and a clean object-oriented design.

## Status

Cited answers work end to end, from the command line and through a web API.

| Area                                                       | Status  |
|------------------------------------------------------------|---------|
| Download the official PDF (checksum verified)              | Done    |
| Parse Laws 1–17 into chunks cited by Law, section and page | Done    |
| Local embeddings stored in Postgres with pgvector          | Done    |
| Hybrid search (vector + keyword, Reciprocal Rank Fusion)   | Done    |
| Retrieval evaluation (hit rate, MRR, mode comparison)      | Done    |
| Answers with citations from an LLM (`ragferee ask`)        | Done    |
| Web API (FastAPI)                                          | Done    |

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

| Variable            | Required                 | Default      | Description        |
|---------------------|--------------------------|--------------|--------------------|
| `POSTGRES_USER`     | yes                      |              | Database user      |
| `POSTGRES_PASSWORD` | yes                      |              | Database password  |
| `POSTGRES_DB`       | no                       | `ragferee`   | Database name      |
| `POSTGRES_HOST`     | no                       | `localhost`  | Database host      |
| `POSTGRES_PORT`     | no                       | `5432`       | Database port      |
| `LLM_PROVIDER`      | no                       | `mock`       | `mock` or `openai` |
| `OPENAI_API_KEY`    | if `LLM_PROVIDER=openai` |              | OpenAI API key     |
| `OPENAI_MODEL`      | no                       | `gpt-5-mini` | OpenAI model       |

The database connection URL is built from these values, so it is not configured separately.

## Usage

Build the search index once (the first run also downloads a 65 MB embedding model):

```bash
uv run ragferee download   # fetch the Laws of the Game PDF into data/raw/
uv run ragferee ingest     # split it into chunks in data/processed/chunks.json
uv run ragferee index      # embed the chunks and store them in Postgres
```

Then ask a question:

```bash
uv run ragferee ask "What happens if a goalkeeper holds the ball too long?"
```

```text
If a goalkeeper inside their penalty area controls the ball with their hand(s)/arm(s) for more than eight
seconds before releasing it, the opposing team is awarded a corner kick [1]. The referee decides when the
goalkeeper's control begins and will visually count down the last five seconds with a raised hand [1]. …

Sources:
[1] Law 12, 3. Corner kick, p. 119
```

Answers use only the retrieved Law text, and every citation comes from a retrieved chunk, so the model cannot
invent a Law, section or page. With the default `LLM_PROVIDER=mock`, `ask` runs without an API key and returns
a placeholder answer with real citations; set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` for real answers.

To see the retrieved chunks themselves, search instead:

```bash
uv run ragferee search "What happens if a goalkeeper holds the ball too long?"
```

```text
1. Law 12, 3. Corner kick, p. 119  (0.0323)
   A corner kick is awarded if a goalkeeper, inside their penalty area, controls the ball with their …
2. Law 12, 1. Direct free kick, p. 116  (0.0320)
   …
```

| Command                             | What it does                                         |
|-------------------------------------|------------------------------------------------------|
| `check`                             | Validate settings without revealing secrets          |
| `download [--force]`                | Download the PDF and verify its checksum             |
| `ingest`                            | Parse the PDF into chunks                            |
| `index`                             | Embed the chunks and store them in Postgres          |
| `search "<question>" [--limit N]`   | Show the most relevant chunks with citations         |
| `ask "<question>" [--limit N]`      | Answer a question, citing Law, section and page      |
| `eval [--compare] [--file PATH]`    | Measure retrieval quality (`--help` for all options) |
| `serve [--host HOST] [--port PORT]` | Run the web API (default `127.0.0.1:8000`)           |

### Web API

```bash
uv run ragferee serve
```

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "Can a player be offside from a throw-in?"}'
```

```json
{
  "text": "No — there is no offside offence if a player receives the ball directly from a throw-in [1]. …",
  "citations": [
    {"law_number": 11, "section": "3. No offence", "page": 111},
    {"law_number": 11, "section": "1. Offside position", "page": 109}
  ]
}
```

| Endpoint      | What it does                                                                     |
|---------------|----------------------------------------------------------------------------------|
| `POST /ask`   | Answer `{"question": "...", "limit": 5}`; `[n]` in the text is `citations[n-1]`  |
| `GET /health` | Report the number of indexed chunks; 503 if the database is down or empty        |
| `GET /docs`   | Interactive API documentation                                                    |

Invalid input returns 422, a failed LLM request 502, and an unavailable database 503.

## How it works

1. **Ingest** – PyMuPDF reads pages 44–151 (Laws 1–17). Laws and sections are detected from font size and text,
   giving one chunk per section per page, so every result can be cited exactly.
2. **Embed** – each chunk is embedded locally with `BAAI/bge-small-en-v1.5` (fastembed, 384 dimensions); no API key needed.
3. **Store** – chunks and embeddings live in Postgres with pgvector; a generated full-text column supports keyword search.
4. **Search** – a question is matched by meaning (cosine distance) and by keywords, and the two rankings are merged
   with weighted Reciprocal Rank Fusion; keyword results count for a quarter (see below).
5. **Answer** – the top 5 chunks are sent to the LLM as numbered sources, with instructions to answer only from
   them and cite each statement as `[n]`. Those numbers are mapped back to the chunks' citations; unknown
   numbers are dropped, and the model replies "not covered" when the sources don't answer the question.
6. **Serve** – the FastAPI app loads the embedding model, LLM client and a Postgres connection pool once at
   startup, then answers each request with the same code as `ragferee ask`.

## Retrieval quality

Retrieval is measured with `uv run ragferee eval --compare` on two sets of hand-checked questions, many phrased in
everyday words such as "shin pads" or "linesman":

- `eval/retrieval.jsonl` – 36 questions covering all 17 Laws, used for tuning
- `eval/retrieval-holdout.jsonl` – 27 different questions, used only to confirm a tuning choice

| Mode                            | Main set (36)<br>Hit@1 / Hit@5 / MRR | Held-out set (27)<br>Hit@1 / Hit@5 / MRR |
|---------------------------------|--------------------------------------|------------------------------------------|
| Hybrid, equal weights (before)  | 0.69 / 0.92 / 0.80                   | 0.52 / 0.85 / 0.64                       |
| **Hybrid, keyword weight 0.25** | **0.75 / 0.97 / 0.83**               | **0.56 / 0.89 / 0.70**                   |
| Vector only                     | 0.64 / 0.97 / 0.77                   | 0.67 / 0.96 / 0.77                       |
| Keyword only                    | 0.50 / 0.72 / 0.59                   | 0.48 / 0.67 / 0.56                       |

Keyword search uses OR matching, so it matches most chunks (a median of 84 of 135 per question) and ranks them
noisily. With equal weights it pushed correct sections out of the top 5. Counting keyword results for a quarter,
and taking 50 candidates from each search instead of 20, improved hybrid search on both sets. The weight was chosen
on the main set for the best hit@5, since all top-5 chunks reach the LLM. Across all 63 questions, hybrid and
vector search now perform about the same: hybrid places the right section first slightly more often, and vector
search finds it in the top 5 slightly more often.

Tried, but didn't help:

- **Law and section header in the embedded text** – improved vector search on the main set (MRR 0.77 → 0.82) but
  made it worse on the held-out set (0.77 → 0.72): chunks from the same Law started to look alike. Reverted.
- **Cover-density ranking (`ts_rank_cd`)** – keyword hit@5 on the main set fell from 0.72 to 0.42.
- **Section and Law titles weighted in the keyword index** – keyword hit@5 fell to 0.53.
- **Length-normalised `ts_rank`** – better for keyword search alone (hit@5 0.83), no gain in hybrid search.

Known weak spots: everyday words that the Laws don't use ("linesman" for assistant referee, "game" for match),
and tables in the PDF, such as pitch dimensions, which extract poorly.

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
