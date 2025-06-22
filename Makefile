.PHONY: help install install-dev build clean test lint format publish test-publish

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package in current environment"
	@echo "  install-dev  - Install package in development mode"
	@echo "  build        - Build distribution packages"
	@echo "  clean        - Clean up build artifacts"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  publish      - Publish to PyPI"
	@echo "  test-publish - Publish to TestPyPI"

# Install package in current environment
install:
	pip install .

# Install package in development mode
install-dev:
	pip install -e .

# Build distribution packages
build: clean
	python setup.py sdist bdist_wheel

# Clean up build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Run tests (if test files exist)
test:
	@if [ -d "tests" ]; then \
		python -m pytest tests/ -v; \
	else \
		echo "Running basic functionality test..."; \
		python -c "import jsontables; print('✓ Package imports successfully')"; \
		echo "✓ Basic import test passed"; \
	fi

# Run linting checks
lint:
	@command -v flake8 >/dev/null 2>&1 || { echo "Installing flake8..."; pip install flake8; }
	flake8 jsontables/ --max-line-length=100 --ignore=E203,W503

# Format code with black
format:
	@command -v black >/dev/null 2>&1 || { echo "Installing black..."; pip install black; }
	black jsontables/

# Publish to PyPI
publish: build
	@command -v twine >/dev/null 2>&1 || { echo "Installing twine..."; pip install twine; }
	twine upload dist/*

# Publish to TestPyPI
test-publish: build
	@command -v twine >/dev/null 2>&1 || { echo "Installing twine..."; pip install twine; }
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Quick demo
demo:
	@echo "Running JSON-Tables demo..."
	@echo '[{"name": "Alice", "age": 30, "score": 95}, {"name": "Bob", "age": 25, "score": 87}]' | python -m jsontables.cli 