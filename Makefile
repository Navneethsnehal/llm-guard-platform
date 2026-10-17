PYTHON ?= python
RUFF ?= ruff
MYPY ?= mypy

.PHONY: install lint format typecheck test redteam eval bench ci

install:
	$(PYTHON) -m pip install -e ".[dev,ui]"

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) check --fix .
	$(RUFF) format .

typecheck:
	$(MYPY)

test:
	$(PYTHON) -m pytest

redteam:
	$(PYTHON) -m llm_guard.cli redteam --smoke --output var/reports/redteam.json

eval:
	$(PYTHON) -m llm_guard.cli eval-gate --smoke --output var/reports/eval.json

bench:
	$(PYTHON) -m llm_guard.cli bench --requests 400 --concurrency 20 --output var/reports/bench.json

ci: lint typecheck test redteam eval
