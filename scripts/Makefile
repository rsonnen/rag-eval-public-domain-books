.PHONY: help format lint typecheck security all clean install-dev sync update-lock

help:
	@echo "Public Domain Books Corpus Scripts - Available targets:"
	@echo ""
	@echo "Quality checks:"
	@echo "  make format       - Run ruff format"
	@echo "  make lint         - Run ruff check with auto-fix"
	@echo "  make typecheck    - Run mypy type checking"
	@echo "  make security     - Run bandit security scanning"
	@echo "  make all          - Run format, lint, security, typecheck"
	@echo ""
	@echo "Dependency management (using uv):"
	@echo "  make install-dev  - Install all dependencies including dev group"
	@echo "  make sync         - Sync dependencies from uv.lock"
	@echo "  make update-lock  - Update uv.lock with latest compatible versions"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Clean build artifacts"

format:
	@echo "Running ruff format..."
	uv run ruff format .

lint:
	@echo "Running ruff check with auto-fix..."
	uv run ruff check --fix .

typecheck:
	@echo "Running mypy type checking..."
	uv run mypy download_books.py build_gutenberg.py build_archive.py

security:
	@echo "Running security checks with bandit..."
	uv run bandit -r . -f screen -ll --exclude ./.venv

all: format lint security typecheck
	@echo "All quality checks passed!"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

install-dev:
	@echo "Installing development dependencies..."
	uv sync --dev
	@echo "Installation complete!"

sync:
	@echo "Syncing dependencies from uv.lock..."
	uv sync

update-lock:
	@echo "Updating dependency lock file..."
	uv lock --upgrade
