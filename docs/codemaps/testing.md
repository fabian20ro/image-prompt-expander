# Testing Codemap

## Test Layout

- Config/settings: `tests/test_config.py`
- Grammar generation + cleanup: `tests/test_grammar_generator.py`
- Tracery runner: `tests/test_tracery_runner.py`
- Image generation wrapper: `tests/test_image_generator.py`
- Gallery HTML generation: `tests/test_gallery.py`, `tests/test_gallery_index.py`
- Gallery service: `tests/test_gallery_service.py`
- Metadata manager: `tests/test_metadata_manager.py`
- Pipeline orchestration: `tests/test_pipeline.py`
- Server routes/models/queue/worker: `tests/test_routes.py`, `tests/test_models.py`, `tests/test_queue_manager.py`, `tests/test_worker.py`, `tests/test_worker_subprocess.py`

## Common Fixtures

- Shared fixtures live in `tests/conftest.py`.
- Most filesystem tests use temporary directories and synthetic run files.

## Notes

- In this environment, `tests/test_image_generator.py` aborts due native MLX/mflux import behavior during one unsupported-model test path.
- Other suites run clean when executed individually with `./venv/bin/pytest`.
