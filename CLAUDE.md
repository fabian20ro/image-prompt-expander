# CLAUDE.md

## Quick Reference

```bash
# Activate venv (required before all commands)
source venv/bin/activate

# Run tests (ALWAYS run after code changes)
pytest -v --tb=short

# Run single test
pytest tests/test_file.py::test_name -v

# Start web server
python src/cli.py --serve

# Generate prompts + images
python src/cli.py -p "prompt" -n 5 --generate-images --prefix name
```

## External Dependencies

- **LM Studio** must be running at `localhost:1234` for grammar generation
- **mflux** only works on Apple Silicon (M1/M2/M3/M4)
- Mock these services in tests - never call real APIs

## Code Patterns

- Use `MetadataManager` for all run metadata operations (not raw JSON loading)
- Use `PipelineConfig` dataclasses for complex parameter passing
- Prefer pathlib over os.path for all file operations
- Use dataclasses or Pydantic models for structured data
- Type hints required for all function signatures

## Testing

- **IMPORTANT**: Run `pytest -v --tb=short` after any code changes
- New features require tests before merge
- Mock external services (LM Studio, mflux) - see existing test patterns
- Use `temp_dir` fixture for file system tests
- Tests use pytest-asyncio for async code

## Gotchas

- Image enhancement replaces originals in-place (creates backup first)
- Queue state persists in `generated/queue.json` - delete to reset
- SSE connections require proper cleanup in error handlers
- Tiled VAE is enabled by default (disable with `--no-tiled-vae`)
- `routes.py` uses FastAPI `Depends()` with `lru_cache` for singletons

## Git Conventions

- Commit messages: imperative mood, explain the "why"
- Keep commits focused on single logical changes
- Run tests before committing

## Architecture Notes

Key modules:
- `pipeline.py` - Core orchestration (use `PipelineExecutor`)
- `metadata_manager.py` - Centralized metadata operations
- `server/routes.py` - FastAPI endpoints with dependency injection
- `server/worker.py` + `worker_subprocess.py` - Background task processing

See @README.md for full project overview.
