# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**image-prompt-expander** - A procedural image prompt generator that creates varied prompts for FLUX.2 image models, with optional local image generation using mflux.

**Pipeline:** User prompt → LLM generates Tracery grammar → Tracery produces N prompts → (optional) mflux generates images

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

# Preview grammar without generating files
python src/cli.py -p "description" --dry-run

# Clean all generated files
python src/cli.py --clean
```

## Architecture

### Three-Stage Pipeline

1. **Grammar Generation** (`src/grammar_generator.py`)
   - Sends user prompt to LM Studio with system prompt from `templates/system_prompt.txt`
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

### Key Files

- `src/cli.py` - Click-based CLI, orchestrates the pipeline
- `templates/system_prompt.txt` - Instructions for LLM to generate Tracery grammars
- `generated/grammars/` - Cached grammars (by prompt hash)
- `generated/prompts/` - Output directories with prompts, images, and metadata

### Output Naming Convention

Files use prefix naming: `{prefix}_{prompt_index}.txt` for prompts, `{prefix}_{prompt_index}_{image_index}.png` for images.

## Dependencies

- LM Studio must be running locally at `http://localhost:1234` for grammar generation
- mflux requires macOS with Apple Silicon (M1/M2/M3/M4) for image generation
