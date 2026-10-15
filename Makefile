.PHONY: install lint format test db-up db-down

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

db-up:
	docker compose up -d

db-down:
	docker compose down
