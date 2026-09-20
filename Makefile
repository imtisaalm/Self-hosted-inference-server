.PHONY: test lint typecheck check

test:
	pytest -q

lint:
	ruff check src scripts tests
	shellcheck scripts/*.sh

typecheck:
	mypy src scripts

check: lint typecheck test
