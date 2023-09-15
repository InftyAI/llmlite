.PHONY: lint
lint:
	mypy ./ --exclude tmp
	black .

.PHONY: test
test: lint
	python -m unittest

.PHONY: integration-test
integration-test: lint
	python -m unittest discover -p "integration*.py"

.PHONY: check
check: lint test integration-test

.PHONY: build
build: lint
	poetry build

.PHONY: publish
publish: build  export-requirements
	poetry publish --username=__token__ --password=$(PYPI_TOKEN)

.PHONEY: export-requirements
export-requirements: export-requirements-dev
	poetry export -f requirements.txt -o requirements.txt --without-hashes

export-requirements-dev:
	poetry export -f requirements.txt -o requirements-dev.txt --without-hashes --with dev
