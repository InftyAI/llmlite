.PHONY: validation
validation:
	mypy ./ --exclude tmp

.PHONY: unit-test
unit-test:
	python -m unittest

.PHONY: e2e-test
e2e-test:
	python -m unittest discover -p "e2e*.py"
