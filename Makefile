.PHONY: install lint format test hooks db-up db-down

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
