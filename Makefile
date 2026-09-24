PYTHON=python
VERSION=patch

.PHONY=test publish bump clean


test:
	pytest

publish: test
	${PYTHON} -m build
	twine upload dist/*

bump: test
	bump-my-version ${VERSION}
	git commit --amend

clean:
	rm -rf .coverage dist build .pytest_cache
