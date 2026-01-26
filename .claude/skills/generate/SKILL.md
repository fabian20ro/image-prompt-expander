---
name: generate
description: Generate images from a prompt using the full pipeline
allowed-tools:
  - Bash(python src/cli.py:*)
  - Bash(source venv/bin/activate:*)
---

# Image Generator

Generate images using the full pipeline: prompt -> LLM -> grammar -> tracery -> images.

## Usage

- `/generate "prompt"` - Generate with default settings
- `/generate "prompt" -n 10 --prefix name` - With options

## Steps

1. Activate venv: `source venv/bin/activate`
2. Run: `python src/cli.py -p $ARGUMENTS --generate-images`
3. Report gallery location when complete

## Common Options

```bash
# Basic generation (5 prompts, 1 image each)
python src/cli.py -p "a dragon" -n 5 --generate-images --prefix dragon

# With enhancement
python src/cli.py -p "a cat" -n 3 --generate-images --enhance --prefix cat

# Memory-efficient enhancement (load models separately)
python src/cli.py -p "a cat" -n 10 --generate-images --enhance --enhance-after --prefix cat

# Multiple images per prompt
python src/cli.py -p "a forest" -n 3 --generate-images --images-per-prompt 2 --prefix forest

# Resume interrupted generation
python src/cli.py --from-prompts generated/prompts/... --generate-images --resume
```

## Requirements

- LM Studio running at localhost:1234
- Apple Silicon Mac for image generation (mflux)