---
project: image-prompt-expander
plan_id: initial-setup-and-testing
status: pending
created_at: 2026-05-20T21:30:00Z

description: |
  Formalize the exploration findings by setting up a robust testing environment and documenting core architecture patterns.

tasks:
  - id: setup-pytest
    tier: 0
    description: Install pytest and configure it for use with `uv run`.
    verification: `uv run pytest --version`

  - id: integration-test-pipeline
    tier: 1
    description: |
      Create a test in `tests/test_pipeline.py` that mocks `generate_grammar` and `run_tracery` to verify the `PipelineExecutor` logic flows correctly from grammar to image directory creation.
    verification: `uv run pytest tests/test_pipeline.py`

  - id: docs-metadata-manager
    tier: 2
    description: |
      Add a developer note in `docs/architecture.md` (create if needed) explaining the `MetadataManager` pattern to avoid direct JSON manipulation.
    verification: Check existence of `docs/architecture.md`.

  - id: testing-infrastructure-check
    tier: 1
    description: Verify that all tests pass in a clean environment.
    verification: `uv run pytest`
---

# Plan: Initial Setup and Testing
Status: **PENDING**

## Overview
Following the initial codebase exploration, this plan focuses on stabilizing the development workflow through testing and documentation of architectural patterns discovered (e.g., MetadataManager, mflux integration).
