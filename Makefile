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
	rm -rf dist

check-python: ## Format the python code (auto fix)
	# uv tool run isort . --verbose --skip .venv
	# uv tool run black .
	# uv tool run ruff check . --fix
	# uv tool run mypy --follow-untyped-imports src
	# uv tool run pylint --verbose --ignore=.venv .
	uv tool run bandit -r src

check-python-nofix: ## Format the python code (no fix)
	# uv tool run isort . --check --verbose --skip .venv
	# uv tool run black . --check
	# uv tool run ruff check .
	# uv tool run mypy --follow-untyped-imports src
	# uv tool run pylint --verbose --ignore=.venv .
	uv tool run bandit -r src

check-python-security: ## security checks only (no-fix)
	uv tool run bandit -r src

black: ## Run black
	uv tool run black .

setup-gitleaks: ## Grab the docker image
	docker pull zricethezav/gitleaks:latest

run-gitleaks: ## run gitleaks with docker
	docker run -v $(CURDIR):/path zricethezav/gitleaks:latest detect --source="/path" --verbose

setup-git-hooks: ## build & add pre-commit and pre-push hooks
	pre-commit install --hook-type pre-commit --hook-type pre-push

build-package: ## Build the package as an installable file locally
	uv build