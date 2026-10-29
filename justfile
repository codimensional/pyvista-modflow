# List available recipes
default:
    @just --list

# Create the virtual environment with the dev group
sync:
    uv sync --group dev

# Delete the virtual environment, then sync
reset:
    rm -rf .venv build dist
    just sync

# Run the tests, forwarding paths and flags to pytest
test *args:
    uv run pytest --cov {{ args }}

# Run type checking
typecheck:
    uv run mypy

# Run all linters and formatters (pre-commit hooks)
lint:
    uvx pre-commit run --all-files

# Render the README figure into docs/images/
figure:
    uv run python scripts/render_readme.py

# Build the sdist and wheel into dist/
build:
    uv build
