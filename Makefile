EMACS ?= emacs
PYTHON ?= python3

.PHONY: test test-elisp test-python

test: test-elisp test-python

test-elisp:
	$(EMACS) -Q --batch -l test/run.el

test-python:
	cd test && $(PYTHON) -B -m unittest -v \
		test_composition test_decoder test_gpu_worker test_japanese test_rescorer test_transducer test_schemes
