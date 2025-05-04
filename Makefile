
install:
	pip install -U pip setuptools -e .

release:
	tox -e clean
	tox -e build
	tox -e publish -- --repository pypi

.PHONY: release
