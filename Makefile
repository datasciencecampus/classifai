.PHONY: all
all: ## Show the available make targets.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: clean
clean: ## Clean the temporary files.
	rm -rf .ruff_cache

check-python: ## Format the python code (auto fix)
	uv tool run ruff check . --fix
	uv tool run ruff format .
	uv tool run bandit -r src

check-python-nofix: ## Format the python code (no fix)
	uv tool run ruff check .
	# uv tool run bandit -r src

check-python-security: ## security checks only (no-fix)
	# uv tool run bandit -r src

setup-gitleaks: ## Grab the docker image
	docker pull zricethezav/gitleaks:latest

run-gitleaks: ## run gitleaks with docker
	docker run -v $(CURDIR):/path zricethezav/gitleaks:latest detect --source="/path" --verbose

setup-git-hooks: ## build & add pre-commit and pre-push hooks
	pre-commit install --hook-type pre-commit --hook-type pre-push

setup-git-hooks-no-docker: ## build & add pre-commit and pre-push hooks
	pre-commit install --hook-type pre-commit --hook-type pre-push -c ".pre-commit-config-NO-DOCKER.yaml"

build-package: ## Build the package as an installable file locally
	uv build