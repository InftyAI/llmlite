.PHONY: lint
lint:
	mypy .
	black .

.PHONY: test
test: unit-test integration-test

.PHONY: unit-test
unit-test: lint
	pytest tests/unit_tests

.PHONY: integration-test
integration-test: lint
	pytest tests/integration_tests

.PHONY: check
check: lint test integration-test

.PHONY: build
build: lint
	poetry build

.PHONY: publish
publish: build  export-requirements
	poetry publish --username=__token__ --password=$(PYPI_TOKEN)

.PHONEY: export-requirements
export-requirements:
	poetry export -f requirements.txt -o requirements.txt --without-hashes
	poetry export -f requirements.txt -o requirements-dev.txt --without-hashes --with dev
