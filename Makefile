.PHONY: install test lint format typecheck build check

install:
	uv sync

install-all:
	uv sync --all-extras

test:
	uv run pytest tests/unit -v

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src/paretoguard

build:
	uv build

check: lint typecheck test
