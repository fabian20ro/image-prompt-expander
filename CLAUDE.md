# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**image-prompt-expander** - A procedural image prompt generator that creates varied prompts for FLUX.2 image models, with optional local image generation using mflux.

**Pipeline:**
```
Full pipeline:     User prompt → LLM → Grammar → Tracery → Prompts → Images → (Enhancement)
--from-grammar:                        Grammar → Tracery → Prompts → Images → (Enhancement)
--from-prompts:                                            Prompts → Images → (Enhancement)
--enhance-images:                                                              Enhancement
--serve:           Web UI with interactive galleries, queue management, real-time progress
```

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start web UI (recommended for interactive use)
python src/cli.py --serve
# Opens browser at http://localhost:8000 with generation form, gallery browser, and queue management

# Generate text prompts only (requires LM Studio running on localhost:1234)
python src/cli.py -p "a dragon flying over mountains" -n 50

# Generate prompts + images (Apple Silicon only)
python src/cli.py -p "a dragon flying over mountains" -n 5 \
    --generate-images --prefix dragon

# Generate images with SeedVR2 2x enhancement
python src/cli.py -p "a cat" -n 1 --generate-images --enhance --prefix test

# Generate images with enhancement (memory-efficient batch mode)
# Use --enhance-after to defer enhancement until after all images are generated
# This loads only one model at a time, reducing peak memory usage by ~50%
python src/cli.py -p "a cat" -n 10 --generate-images --enhance --enhance-after --prefix test

# Preview grammar without generating files
python src/cli.py -p "description" --dry-run

# Resume from cached grammar (skip LLM generation)
python src/cli.py --from-grammar generated/grammars/abc123.tracery.json \
    -n 100 --prefix dragon2

# Resume from existing prompts (generate images only)
python src/cli.py --from-prompts generated/prompts/abc123_20260124_122208 \
    --generate-images --images-per-prompt 2 --resume

# Standalone enhancement (enhance existing images)
python src/cli.py --enhance-images path/to/image.png
python src/cli.py --enhance-images path/to/folder/
python src/cli.py --enhance-images "generated/prompts/*/test_*.png"

# Generate with live gallery (auto-created before images)
python src/cli.py -p "a cat" -n 5 --generate-images --prefix test
# Gallery URL printed to terminal - open in browser, refresh to see progress

# Resume interrupted generation (skip existing images)
python src/cli.py --from-prompts generated/prompts/... --generate-images --resume
# Also works with full pipeline:
python src/cli.py -p "a cat" -n 10 --generate-images --prefix test --resume

# Clean all generated files
python src/cli.py --clean
```

## Architecture

### Five-Stage Pipeline

1. **Grammar Generation** (`src/grammar_generator.py`)
   - Sends user prompt to LM Studio with model-specific system prompt
   - Uses `templates/system_prompt_z-image-turbo.txt` (camera-first) or `templates/system_prompt_flux2-klein.txt` (prose-based)
   - LLM returns a Tracery grammar (JSON) that locks specified elements and varies unspecified ones
   - Grammars are cached by prompt hash in `generated/grammars/`

2. **Prompt Expansion** (`src/tracery_runner.py`)
   - Expands the Tracery grammar N times to produce unique prompt variations
   - Uses `tracery` library with base English modifiers

3. **Image Generation** (`src/image_generator.py`)
   - Optional: renders prompts to images using mflux (MLX-based FLUX for Apple Silicon)
   - Supports z-image-turbo, flux2-klein-4b, flux2-klein-9b models
   - Uses pre-quantized 4-bit models from HuggingFace when available
   - Model instances are cached to avoid reloading between images
   - Tiled VAE decoding enabled by default (reduces memory, disable with `--no-tiled-vae`)

4. **Image Enhancement** (`src/image_enhancer.py`)
   - Optional: enhances images in-place using SeedVR2 with 2x upscaling (replaces originals)
   - Configurable softness parameter (0.0-1.0)
   - Supports standalone mode for enhancing existing images
   - Model instances are cached to avoid reloading between images
   - Tiled VAE decoding enabled by default (reduces memory, disable with `--no-tiled-vae`)

5. **Gallery Generation** (`src/gallery.py`, `src/gallery_index.py`)
   - Creates interactive HTML galleries via the web UI (`--serve`)
   - Shows prompts below each image (scrollable for long prompts)
   - Editable Tracery grammar with save/regenerate functionality
   - Per-image generate/enhance buttons
   - Links to raw LLM response file
   - Real-time updates via SSE as images are generated
   - Auto-generates master index with generation form at `generated/index.html`

### Web UI Server (`src/server/`)

FastAPI-based web interface for the generation pipeline:

- `app.py` - FastAPI application with lifespan handling and static file serving
- `routes.py` - API endpoints for generation, queue management, and galleries
- `queue_manager.py` - Disk-based queue persistence (`generated/queue.json`)
- `worker.py` - Background task processor with subprocess spawning
- `worker_subprocess.py` - Isolated execution script for heavy operations
- `models.py` - Pydantic models for API requests/responses

**API Endpoints**:
- `GET /index` - Master index with generation form
- `GET /gallery/{run_id}` - Interactive gallery page
- `POST /api/generate` - Start new generation pipeline
- `GET /api/events` - SSE stream for real-time updates
- `POST /api/queue/clear` - Clear pending tasks
- `POST /api/worker/kill` - Kill current task
- `PUT /api/gallery/{id}/grammar` - Update grammar
- `POST /api/gallery/{id}/regenerate` - Regenerate prompts

### Key Files

- `src/cli.py` - Click-based CLI, orchestrates the pipeline
- `src/server/` - Web UI server package (FastAPI + SSE)
- `src/image_enhancer.py` - SeedVR2 image enhancement module
- `src/gallery.py` - HTML gallery generation with live updates
- `src/gallery_index.py` - Master index generation linking all galleries
- `templates/system_prompt.txt` - Default instructions for LLM to generate Tracery grammars
- `templates/system_prompt_z-image-turbo.txt` - Camera-first prompt structure for z-image-turbo
- `templates/system_prompt_flux2-klein.txt` - Prose-based prompt structure for flux2-klein models
- `generated/index.html` - Master index linking all run galleries
- `generated/grammars/` - Cached grammars (by prompt hash)
- `generated/prompts/` - Output directories with prompts, images, and metadata
- `generated/queue.json` - Task queue persistence for web UI

### Output Naming Convention

Files use prefix naming:
- `{prefix}_{prompt_index}.txt` for prompts
- `{prefix}_{prompt_index}_{image_index}.png` for images (enhanced in-place if `--enhance` used)
- `{prefix}_gallery.html` for the live-updating image gallery
- `{prefix}_grammar.json` for the Tracery grammar
- `{prefix}_raw_response.txt` for the raw LLM response (with thinking blocks)
- `{prefix}_metadata.json` for generation settings

## Testing

Run tests after making changes to verify functionality:

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_grammar_generator.py

# Run tests with coverage (if pytest-cov installed)
pytest --cov=src
```

## Dependencies

- LM Studio must be running locally at `http://localhost:1234` for grammar generation
- mflux requires macOS with Apple Silicon (M1/M2/M3/M4) for image generation
