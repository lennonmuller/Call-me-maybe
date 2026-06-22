UV := $(HOME)/.local/bin/uv
SGOINFRE := $(shell [ -d /sgoinfre/$(USER) ] && echo /sgoinfre/$(USER) || echo $(HOME))
UV_ENV := UV_CACHE_DIR=$(SGOINFRE)/.cache/uv UV_PROJECT_ENVIRONMENT=$(SGOINFRE)/.venv_cmm HF_HOME=$(SGOINFRE)/.cache/huggingface


.PHONY: install run debug clean lint lint-strict test

install:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	mkdir -p $(SGOINFRE)/.cache/uv
	mkdir -p $(SGOINFRE)/.venv_cmm
	mkdir -p $(SGOINFRE)/.cache/huggingface
	$(UV_ENV) $(UV) sync

run:
	$(UV_ENV) $(UV) run python -m src

debug:
	$(UV_ENV) $(UV) run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache src/__pycache__
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf data/output/*

lint:
	$(UV_ENV) $(UV) run flake8 .
	$(UV_ENV) $(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

test:
	$(UV_ENV) $(UV) run pytest -q
