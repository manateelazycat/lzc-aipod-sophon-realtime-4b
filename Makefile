PYTHON ?= python3

.PHONY: test lint
test:
	PYTHONPATH=runtime:. $(PYTHON) -m pytest -q tests

lint:
	lzc-cli project lint
