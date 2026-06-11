PYTHON=python

.PHONY=test clean

test:
	pytest

clean:
	rm -rf .coverage dist build *.egg-info
