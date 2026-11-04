UV ?= uv
UV_RUN := $(UV) run --group examples

EXAMPLE_SRCS := $(wildcard examples/*/[0-9]*.py) $(wildcard examples/*/cookbook/*.py)
EXAMPLE_NBS := $(EXAMPLE_SRCS:.py=.ipynb)

.PHONY: test lint format examples clean-examples uv.lock

# Regenerate the lockfile from the standard index, ignoring any local uv
# configuration (e.g. an organization package mirror).
uv.lock:
	$(UV) lock --no-config --default-index https://pypi.org/simple

# Run the test suite; first make sure the committed lockfile references only
# the standard index (a mirror-generated uv.lock must not be committed).
test:
	@bad="$$(grep -oE 'https://[^"]+' uv.lock | grep -vE '^https://(pypi\.org|files\.pythonhosted\.org)/' | sort -u)"; \
	if [ -n "$$bad" ]; then \
		echo "uv.lock references non-standard package indexes:"; \
		echo "$$bad"; \
		echo "run 'make uv.lock' to regenerate it from pypi.org"; \
		exit 1; \
	fi
	$(UV) run pytest --cov

lint:
	$(UV) run ruff format --check .
	$(UV) run ruff check .
	$(UV) run mypy
	$(UV) run lint-imports

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

# Convert each example script to a Jupyter notebook and execute it in place.
examples: $(EXAMPLE_NBS)

%.ipynb: %.py
	$(UV_RUN) ipynb-py-convert $< $@
	$(UV_RUN) jupyter nbconvert --to notebook --execute --inplace $@

clean-examples:
	rm -f $(EXAMPLE_NBS)
