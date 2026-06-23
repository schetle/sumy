PYTHON=python
VERSION=patch

.PHONY=test publish bump clean


test:
	pytest tests/

publish: test
	${PYTHON} -m build
	twine upload dist/*

bump: test
	bump-my-version bump ${VERSION}

clean:
	rm -rf .bumpversion.cfg .coverage dist build
