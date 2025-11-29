#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = IReranker
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER ?= python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies (defaults to tests+lint)
.PHONY: requirements requirements-test requirements-dev
requirements: requirements-test

requirements-test:
	pip install -e ".[tests,lint]"

requirements-dev:
	pip install -e ".[dev,beir]"
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## (dataset rule removed)


## Evaluate all registered rankers on BEIR datasets (config-driven)
.PHONY: beir-eval
beir-eval:
	$(PYTHON_INTERPRETER) -m ireranker.run_beir_eval $(ARGS)
 


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
