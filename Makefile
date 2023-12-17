.PHONY: lint
lint:
	# mypy .
	black .

.PHONY: test
test: unit-test integration-test

.PHONY: unit-test
unit-test: lint
ifdef key
	pytest tests/unit_tests -v -k $(key)
else
	pytest tests/unit_tests
endif

.PHONY: integration-test
integration-test: lint
ifdef key
	pytest tests/integration_tests -v -k $(key)
else
	pytest tests/integration_tests
endif

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
