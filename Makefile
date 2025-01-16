test:
	poetry run pytest -n auto --cov=pycco -vv tests/ --doctest-modules pycco --without-integration --without-slow-integration ${PYTEST_ARGS}

check:
	poetry run pre-commit run --all-files

lint:
	make check

all: lint test

dev-release:
	hatch version dev
	hatch build
	hatch publish