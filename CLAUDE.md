# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**image-prompt-expander** - A procedural image prompt generator that creates varied prompts for FLUX.2 image models, with optional local image generation using mflux.

**Pipeline:**
```
Full pipeline:     User prompt → LLM → Grammar → Tracery → Prompts → Gallery → Images → (Enhancement)
--from-grammar:                        Grammar → Tracery → Prompts → Gallery → Images → (Enhancement)
--from-prompts:                                            Prompts → Gallery → Images → (Enhancement)
--enhance-images:                                                                        Enhancement
--gallery:                                                           Gallery
```

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate text prompts only (requires LM Studio running on localhost:1234)
python src/cli.py -p "a dragon flying over mountains" -n 50

# Generate prompts + images (Apple Silicon only)
python src/cli.py -p "a dragon flying over mountains" -n 5 \
    --generate-images --prefix dragon

# Generate images with SeedVR2 2x enhancement
python src/cli.py -p "a cat" -n 1 --generate-images --enhance --prefix test

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

# Generate gallery for existing output directory
python src/cli.py --gallery generated/prompts/20260125_143022_abc123

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

4. **Image Enhancement** (`src/image_enhancer.py`)
   - Optional: enhances images in-place using SeedVR2 with 2x upscaling (replaces originals)
   - Configurable softness parameter (0.0-1.0)
   - Supports standalone mode for enhancing existing images
   - Model instances are cached to avoid reloading between images

5. **Gallery Generation** (`src/gallery.py`)
   - Creates live-updating HTML gallery with image grid
   - Shows prompts below each image
   - Placeholders for pending images, updated as each completes
   - Supports standalone mode via `--gallery` flag

### Key Files

- `src/cli.py` - Click-based CLI, orchestrates the pipeline
- `src/image_enhancer.py` - SeedVR2 image enhancement module
- `src/gallery.py` - HTML gallery generation with live updates
- `templates/system_prompt.txt` - Default instructions for LLM to generate Tracery grammars
- `templates/system_prompt_z-image-turbo.txt` - Camera-first prompt structure for z-image-turbo
- `templates/system_prompt_flux2-klein.txt` - Prose-based prompt structure for flux2-klein models
- `generated/grammars/` - Cached grammars (by prompt hash)
- `generated/prompts/` - Output directories with prompts, images, and metadata

### Output Naming Convention

Files use prefix naming:
- `{prefix}_{prompt_index}.txt` for prompts
- `{prefix}_{prompt_index}_{image_index}.png` for images (enhanced in-place if `--enhance` used)
- `{prefix}_gallery.html` for the live-updating image gallery

## Dependencies

- LM Studio must be running locally at `http://localhost:1234` for grammar generation
- mflux requires macOS with Apple Silicon (M1/M2/M3/M4) for image generation
