# CLAUDE.md

## Quick Start

```bash
# Activate venv (required in each shell session)
source venv/bin/activate

# Run all tests
./venv/bin/pytest -v --tb=short

# Start web UI
python src/cli.py --serve

# Generate prompts + images (CLI)
python src/cli.py -p "prompt" -n 5 --generate-images --prefix name
```

## Core Rules

- Run tests after code changes.
- Mock external services in tests (LM Studio, mflux).
- Use `pathlib.Path` and typed interfaces (`dataclass`/Pydantic) for structured code.
- Prefer `MetadataManager` for metadata operations.
- Keep commits focused and explain the "why" in commit messages.

## Mandatory Session Checklist

1. Review and update `LESSONS_LEARNED.md` during every session.
2. Add at least one brief, concrete lesson when you find behavior that did not work as expected.

## External Runtime Dependencies

- LM Studio at `http://localhost:1234/v1` for grammar generation.
- `mflux` for image generation/enhancement (Apple Silicon).

## Project Maps And Deeper Docs

- Full usage + feature docs: `README.md`
- Code map index: `docs/codemaps/README.md`
- Pipeline map: `docs/codemaps/pipeline.md`
- Server + UI map: `docs/codemaps/server-ui.md`
- Tests + coverage map: `docs/codemaps/testing.md`
