.PHONY: install lint format typecheck
install:
	poetry install --no-root
lint:
	poetry run ruff check src
format:
	poetry run ruff format .
typecheck:
	poetry run mypy src