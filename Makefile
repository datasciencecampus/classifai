.PHONY: all
all: ## Show the available make targets.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: clean
clean: ## Clean the temporary files.
	rm -rf .mypy_cache
	rm -rf .ruff_cache	

check-python: ## Format the python code (auto fix)
	poetry run isort . --verbose
	poetry run black .
	poetry run ruff check . --fix
	poetry run mypy --follow-untyped-imports src
	poetry run pylint --verbose .
	poetry run bandit -r src/classifAI_API

check-python-nofix: ## Format the python code (no fix)
	#poetry run isort . --check --verbose
	#poetry run black . --check
	#poetry run ruff check .
	#poetry run mypy --follow-untyped-imports src
	#poetry run pylint --verbose .
	#poetry run bandit -r src/classifAI_API

black: ## Run black
	poetry run black .

setup-gitleaks: ## Grab the docker image
	docker pull zricethezav/gitleaks:latest

run-gitleaks: ## run gitleaks with docker
	docker run -v $(CURDIR):/path zricethezav/gitleaks:latest detect --source="/path" --verbose