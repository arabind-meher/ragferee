.PHONY: install lint format test hooks db-up db-down setup run

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

hooks:
	uv run pre-commit run --all-files

db-up:
	docker compose up -d

db-down:
	docker compose down

# Download the Laws PDF, split it into chunks and index them (safe to run again)
setup:
	uv run ragferee download
	uv run ragferee ingest
	uv run ragferee index

# Start the API in the background and the UI in front; Ctrl+C stops both
run:
	uv run ragferee serve & api=$$!; \
	trap 'kill $$api 2>/dev/null' INT TERM EXIT; \
	uv run streamlit run src/ragferee/ui.py
