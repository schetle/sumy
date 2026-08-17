PYTHON=python

.PHONY=test publish clean

test:
	pytest

publish: test
	${PYTHON} -m build
	twine upload dist/*

clean:
	rm -rf .coverage dist build *.egg-info
