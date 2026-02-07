# Codemaps

This folder is a fast navigation layer for the codebase.

## At A Glance

- CLI entrypoint: `src/cli.py`
- Pipeline orchestration: `src/pipeline.py`
- Grammar generation: `src/grammar_generator.py`
- Tracery expansion: `src/tracery_runner.py`
- Image generation/enhancement: `src/image_generator.py`, `src/image_enhancer.py`
- Gallery/index HTML generation: `src/gallery.py`, `src/gallery_index.py`, `src/html_components.py`
- Web server/API: `src/server/app.py`, `src/server/routes.py`, `src/server/models.py`
- Task queue + worker: `src/server/queue_manager.py`, `src/server/worker.py`, `src/server/worker_subprocess.py`
- Shared utilities: `src/utils.py`, `src/config.py`, `src/metadata_manager.py`

## Runtime Modes

1. CLI mode (`python src/cli.py ...`)
2. Web mode (`python src/cli.py --serve`)
3. Worker subprocess mode (`src/server/worker_subprocess.py`)

## Data Roots

- Active outputs: `generated/prompts/`
- Cached grammars: `generated/grammars/`
- Flat archives: `generated/saved/`
- Queue persistence: `generated/queue.json`

## Detailed Maps

- Pipeline flow: `docs/codemaps/pipeline.md`
- Server/UI flow: `docs/codemaps/server-ui.md`
- Test map: `docs/codemaps/testing.md`
