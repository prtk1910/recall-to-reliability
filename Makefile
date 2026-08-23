SHELL := /bin/sh
PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/src

ifneq (,$(wildcard .env))
include .env
export OPENAI_API_KEY
endif

.PHONY: paper test generate smoke clean

paper:
	@test -n "$$OPENAI_API_KEY" || { echo "OPENAI_API_KEY is required" >&2; exit 2; }
	$(PYTHON) -m reliabmem.pipeline paper

test:
	$(PYTHON) -m unittest discover -s tests -v

generate:
	$(PYTHON) -m reliabmem.pipeline generate

smoke:
	@test -n "$$OPENAI_API_KEY" || { echo "OPENAI_API_KEY is required" >&2; exit 2; }
	$(PYTHON) -m reliabmem.pipeline smoke

clean:
	$(PYTHON) -m reliabmem.pipeline clean
