EMACS ?= emacs
PYTHON ?= python3

.PHONY: test test-elisp test-python

test: test-elisp test-python

test-elisp:
	$(EMACS) -Q --batch -l test/run.el

test-python:
	$(PYTHON) -B test/run-runtime.py
