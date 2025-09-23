#!/bin/bash
# Script to run tests

echo "Running tests..."
python -m pytest tests/ -v --cov=. --cov-report=html
echo "Tests completed. Coverage report available in htmlcov/index.html"
