---
name: test
description: Run project tests with various options
allowed-tools:
  - Bash(pytest:*)
  - Bash(source venv/bin/activate:*)
  - Bash(./venv/bin/python:*)
---

# Test Runner

Run tests for the image-prompt-expander project.

## Usage

- `/test` - Run all tests
- `/test file.py` - Run specific test file
- `/test -k name` - Run tests matching pattern

## Steps

1. Ensure venv is activated: `source venv/bin/activate`
2. Run pytest with arguments: `pytest $ARGUMENTS -v --tb=short`
3. Report results including:
   - Number of tests passed/failed
   - Any failure details with file:line references
   - Coverage summary if `--cov` was used

## Common Patterns

```bash
# Run all tests
pytest -v --tb=short

# Run specific test file
pytest tests/test_grammar_generator.py -v

# Run tests matching pattern
pytest -k "grammar" -v

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run single test
pytest tests/test_pipeline.py::TestPipelineResult::test_successful_result -v
```