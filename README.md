# Ragferee

Retrieval-augmented Q&A over the IFAB Laws of the Game. Ask a football rules question and get a plain-language explanation backed by the exact rule text, cited by Law, section and page. Built with Python, FastAPI and pgvector, with hybrid retrieval, an evaluation suite, and a clean object-oriented design.

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

| Variable | Required | Default | Description |
|---|---|---|---|
| `POSTGRES_USER` | yes | | Database user |
| `POSTGRES_PASSWORD` | yes | | Database password |
| `POSTGRES_DB` | no | `ragferee` | Database name |
| `POSTGRES_HOST` | no | `localhost` | Database host |
| `POSTGRES_PORT` | no | `5432` | Database port |
| `LLM_PROVIDER` | no | `mock` | `mock` or `openai` |
| `OPENAI_API_KEY` | if `LLM_PROVIDER=openai` | | OpenAI API key |

The database connection URL is built from these values, so it is not configured separately.

## Usage

```bash
uv run ragferee --help
uv run ragferee --version
uv run ragferee check
```

## Development

| Command | What it does |
|---|---|
| `make install` | Install dependencies |
| `make lint` | Ruff lint, Ruff format check and mypy (strict) |
| `make format` | Format and auto-fix lint issues |
| `make test` | Run the test suite |
| `make db-up` / `make db-down` | Start / stop Postgres with pgvector |

Pre-commit hooks run the same checks on every commit:

```bash
uv run pre-commit install
```

## Data

The IFAB Laws of the Game are not included in this repository. Everything under `data/` except the folder structure is ignored by git.

## License

[MIT](LICENSE)
