# CLAUDE.md

## Quick Reference

```bash
# Activate venv (required before all commands)
source venv/bin/activate

# Run tests
pytest -v --tb=short

# Start web server
python src/cli.py --serve

# Generate prompts + images
python src/cli.py -p "prompt" -n 5 --generate-images --prefix name
```

## Dependencies

- LM Studio must be running at `localhost:1234` for grammar generation
- mflux only works on Apple Silicon (M1/M2/M3/M4)

## Code Style

- Use type hints for all function signatures
- Prefer pathlib over os.path
- Use dataclasses or Pydantic models for structured data
- Keep functions under 50 lines

## Gotchas

- Image enhancement replaces originals in-place
- Queue state persists in `generated/queue.json` - delete to reset
- SSE connections require proper cleanup in error handlers
- Tiled VAE is enabled by default (disable with `--no-tiled-vae`)

## Testing

```bash
pytest tests/test_file.py::test_name -v  # Run single test
pytest --cov=src                          # With coverage
```

- New features need tests before merge
- Mock external services (LM Studio, mflux) in tests
