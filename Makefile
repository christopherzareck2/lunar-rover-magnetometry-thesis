# Lunar rover magnetometry thesis code: tests, figures and full reruns from one entry point.
# Run `make help` for the list of targets.
#
# Python: the `lunar-mag` conda environment (environment.yml). Activate it, or let this Makefile find it under
# ~/miniforge3/envs/lunar-mag. Override with `make PYTHON=/path/to/python ...`, or put that in a file named
# local.mk next to this one (not tracked by git).
# Nothing here pushes, publishes or uploads anything.

SHELL := /bin/bash
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(ROOT)/local.mk

ENV_PY := $(HOME)/miniforge3/envs/lunar-mag/bin/python
ifeq ($(CONDA_DEFAULT_ENV),lunar-mag)
  PYTHON ?= python
else ifneq ($(wildcard $(ENV_PY)),)
  PYTHON ?= $(ENV_PY)
else
  PYTHON ?= python
endif
export PYTHONPATH := $(ROOT)/src:$(PYTHONPATH)
export MPLBACKEND := Agg

NICE ?= nice -n 10

.PHONY: help test test-all figures reproduce rerun clean info

help:  ## list the targets
	@echo "Targets (run from the repository root):"
	@awk -F':.*## ' '/^[a-z-]+:.*## /{printf "  make %-10s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

info:  ## print versions, paths and the list of thesis figures
	$(PYTHON) -m lunarmag info

# pytest comes with `conda env update -f environment.yml`. If the env predates that, the test targets borrow the
# base environment's pytest through a temporary path shim in build/pytest-shim (nothing is installed anywhere).
BASE_PY ?= $(HOME)/miniforge3/bin/python
SHIM := $(ROOT)/build/pytest-shim
PYTEST_IN_ENV := $(shell $(PYTHON) -c "import pytest" >/dev/null 2>&1 && echo yes)

define pytest_shim
	@echo "pytest is not installed in $(PYTHON)."
	@echo "  To install it: conda env update -f environment.yml   (or: conda install -n lunar-mag pytest)"
	@echo "  For now, using the base environment's pytest ($(BASE_PY)) through a temporary shim; nothing is installed."
	@SP=$$($(BASE_PY) -c "import os, pytest; print(os.path.dirname(os.path.dirname(pytest.__file__)))" 2>/dev/null) \
	  || { echo "No pytest in $(BASE_PY) either. Run: conda env update -f environment.yml"; exit 1; }; \
	  rm -rf $(SHIM); mkdir -p $(SHIM); \
	  for x in pytest _pytest pluggy iniconfig py.py; do [ -e "$$SP/$$x" ] && ln -s "$$SP/$$x" $(SHIM)/$$x; done; true
endef

test:  ## fast tests, 117 tests in about 40 s: exact reruns of stored numbers, analytic checks, quoted numbers, figures
ifeq ($(PYTEST_IN_ENV),yes)
	$(PYTHON) -m pytest -m "not slow"
else
	$(pytest_shim)
	PYTHONPATH=$(SHIM):$(PYTHONPATH) $(PYTHON) -m pytest -p no:cacheprovider -m "not slow"
endif

test-all:  ## fast tests plus the slow ones (notebook rerun, a09 tilt parts); a few minutes
ifeq ($(PYTEST_IN_ENV),yes)
	$(NICE) $(PYTHON) -m pytest
else
	$(pytest_shim)
	PYTHONPATH=$(SHIM):$(PYTHONPATH) $(NICE) $(PYTHON) -m pytest -p no:cacheprovider
endif

figures:  ## redraw every thesis figure from results/ into thesis/MIT-thesis-template/figures/ and rewrite the manifest
	$(PYTHON) -m lunarmag figures

reproduce:  ## one-report reproduction check from the stored results (about 20 s)
	$(NICE) $(PYTHON) -m lunarmag reproduce

# Full rerun of every simulation (3.5 to 4 hours on an Apple M-series laptop, almost all of it
# analysis/a09_systematics.py). It runs in a scratch copy, so results/ is never overwritten, then compares.
RERUN_DIR := $(ROOT)/build/rerun
RERUN_SCRIPTS := \
  notebook/part1_single_draw.py \
  engine/inversion_approach_comparison.py \
  analysis/a02_isolated_outcrop_paper_geometry.py \
  analysis/a02b_detection_sweep_N1000.py \
  analysis/a04_text_number_checks.py \
  analysis/a06_isolated_robustness.py \
  engine/skatepark_1cm_dr_sweep.py \
  analysis/a08_skatepark_neighbor_directions.py \
  engine/three_layer_wall.py \
  analysis/a03_three_layer_direction_vs_distance.py \
  analysis/a07_three_layer_wall_extent.py \
  engine/scripts/elevated_traverse_height_sweep.py \
  analysis/a05_background_and_correlated_noise.py \
  analysis/a09_systematics.py

rerun:  ## rerun every simulation in a scratch copy and compare with results/ (3.5 to 4 hours)
	@mkdir -p $(RERUN_DIR)
	rsync -a --delete --exclude .git --exclude build --exclude thesis --exclude '__pycache__' $(ROOT)/ $(RERUN_DIR)/
	@cd $(RERUN_DIR) && : > timing.txt && for s in $(RERUN_SCRIPTS); do \
	  args=""; [[ $$s == *part1_single_draw.py ]] && args="--write"; \
	  t0=$$(date +%s); echo "== $$s"; \
	  PYTHONPATH=$(RERUN_DIR)/src $(NICE) $(PYTHON) $$s $$args > logs_$$(basename $$s .py).txt 2>&1 \
	    || { echo "FAILED: $$s (see build/rerun/logs_$$(basename $$s .py).txt)"; exit 1; }; \
	  echo "$$s $$(( $$(date +%s) - t0 )) s" | tee -a timing.txt; done
	$(PYTHON) -m lunarmag compare $(RERUN_DIR)/results $(ROOT)/results

clean:  ## remove build/, caches and __pycache__
	rm -rf $(ROOT)/build .pytest_cache
	find $(ROOT) -name __pycache__ -type d -prune -exec rm -rf {} +
