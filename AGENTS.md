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

## Memory & Continuous Learning

This project maintains a persistent learning system across AI agent sessions.

### Required Workflow

1. **Start of task:** Read `LESSONS_LEARNED.md` before writing any code
2. **During work:** Note any surprises, gotchas, or non-obvious discoveries
3. **End of iteration:** Append to `ITERATION_LOG.md` with what happened
4. **End of iteration:** If the insight is reusable and validated, also add to `LESSONS_LEARNED.md`
5. **Pattern detection:** If the same issue appears 2+ times in the log, promote it to a lesson

### Files

| File | Purpose | When to Write |
|------|---------|---------------|
| `LESSONS_LEARNED.md` | Curated, validated, reusable wisdom | End of iteration (if insight is reusable) |
| `ITERATION_LOG.md` | Raw session journal, append-only | End of every iteration (always) |

### Rules

- Never delete entries from `ITERATION_LOG.md` - it's append-only
- In `LESSONS_LEARNED.md`, obsolete lessons go to the Archive section, not deleted
- Keep entries concise - a future agent scanning 100 entries needs signal, not prose
- Date-stamp everything in `YYYY-MM-DD` format
- When in doubt about whether something is worth logging: log it

## Core Rules

- Run tests after code changes.
- Mock external services in tests (LM Studio, mflux).
- Use `pathlib.Path` and typed interfaces (`dataclass`/Pydantic) for structured code.
- Prefer `MetadataManager` for metadata operations.
- Keep commits focused and explain the "why" in commit messages.

## Mandatory Session Checklist

1. Review `LESSONS_LEARNED.md` at the start of every task.
2. Append an entry to `ITERATION_LOG.md` at the end of every iteration.
3. Add at least one brief, concrete lesson to `LESSONS_LEARNED.md` when behavior did not work as expected and the insight is reusable.

## External Runtime Dependencies

- LM Studio at `http://localhost:1234/v1` for grammar generation.
- `mflux` for image generation/enhancement (Apple Silicon).

## Project Maps And Deeper Docs

- Full usage + feature docs: `README.md`
- Code map index: `docs/codemaps/README.md`
- Pipeline map: `docs/codemaps/pipeline.md`
- Server + UI map: `docs/codemaps/server-ui.md`
- Tests + coverage map: `docs/codemaps/testing.md`
