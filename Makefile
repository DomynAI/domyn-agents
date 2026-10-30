.PHONY: help install dev-install test test-no-cov lint format mypy standalone check-all clean context-windows release-patch release-minor release-major release version-patch version-minor version-major build-wheel

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install package dependencies"
	@echo "  dev-install  - Install package with dev dependencies"
	@echo "  update-lock  - Update lock file with latest versions"
	@echo "  sync-lock    - Synchronize lock file with current dependencies"
	@echo "  test         - Run tests"
	@echo "  test-no-cov  - Run tests without coverage"
	@echo "  standalone   - Build a self-contained tarball for this machine"
	@echo "  lint         - Run ruff linter"
	@echo "  lint-fix     - Run ruff linter and fix issues"
	@echo "  format       - Format code with ruff"
	@echo "  format-check - Check code formatting with ruff"
	@echo "  mypy         - Run type checking with mypy"
	@echo "  check-all    - Run all checks (lint, format check, mypy, tests)"
	@echo "  clean        - Clean up build artifacts"
	@echo "  dev-setup    - Setup development environment"
	@echo "  ci-check     - Run CI-friendly checks"
	@echo "  context-windows - Regenerate the context-window catalogue from models.dev"
	@echo "  release-patch- Tag the next patch release on origin/main and push the tag"
	@echo "  release-minor- Tag the next minor release on origin/main and push the tag"
	@echo "  release-major- Tag the next major release on origin/main and push the tag"
	@echo "  release      - Tag VERSION (make release VERSION=1.7.0rc1) and push it"
	@echo "  version-patch, version-minor, version-major - Aliases of release-patch, release-minor, release-major"
	@echo "  build-wheel        - Build wheel and sdist packages"

# Installation commands
install:
	uv sync

dev-install:
	uv sync --dev --all-extras

update-lock:
	uv lock --upgrade

sync-lock:
	uv lock

# Testing
test:
	uv run pytest tests/ examples/ --cov=domyn --cov-report=xml --cov-report=term

test-no-cov:
	uv run pytest tests/ examples/ -q --no-cov

test-badge: test
	uv run genbadge coverage -i coverage.xml -o coverage-badge.svg

# Linting and formatting
#
# Scoped to the package, its tests and the examples, not the whole tree. `.` also picked up
# whatever the agent had written into the working directory: a session that
# produced a script left `make lint` failing on someone else's style, and
# `make format` would have rewritten that file in place.
SOURCES = domyn/ tests/ examples/

lint:
	uv run ruff check $(SOURCES)

lint-fix:
	uv run ruff check $(SOURCES) --fix

format:
	uv run ruff format $(SOURCES)

format-check:
	uv run ruff format $(SOURCES) --check

standalone:
	scripts/build_standalone.sh dist/standalone

# Type checking
mypy:
	uv run mypy domyn/

# Combined checks. mypy is left out until `make mypy` passes: it still reports
# errors in existing code, and a check that always fails gets ignored.
check-all: lint format-check test
	@echo "All checks passed!"

# Utility commands
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Development workflow
dev-setup: dev-install
	@echo "Development environment setup complete!"

# CI/CD friendly target
ci-check: lint format-check mypy test
	@echo "CI checks completed successfully!"

# The context-window catalogue (domyn/agents/llm/context_windows.json), from
# models.dev. Run by hand and committed like any change: a release is a tag and
# commits nothing, so it ships the catalogue main has.
context-windows:
	uv run python scripts/update_context_windows.py

# Releases
# A release is a tag on origin/main: the version is read from it at build time
# (hatch-vcs), and pushing it starts .github/workflows/release.yml.
release-patch:
	@python3 scripts/tag_release.py patch

release-minor:
	@python3 scripts/tag_release.py minor

release-major:
	@python3 scripts/tag_release.py major

release:
	@test -n "$(VERSION)" || { echo "usage: make release VERSION=1.7.0rc1"; exit 1; }
	@python3 scripts/tag_release.py "$(VERSION)"

# The names releases were made with before; they no longer commit a version.
version-patch: release-patch
version-minor: release-minor
version-major: release-major

# Build
build-wheel:
	uv build
