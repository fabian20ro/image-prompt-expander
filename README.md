# image-prompt-expander

**[View Live Demo](https://fabian20ro.github.io/image-prompt-expander/)** — See example outputs from the generator

A procedural image prompt generator that creates varied, high-quality prompts for FLUX.2 image models, with optional local image generation using mflux.

```
User prompt → LLM generates Tracery grammar → Tracery produces N prompts → (optional) mflux generates images
```

## Requirements

**System:**
- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3/M4) for image generation
- [LM Studio](https://lmstudio.ai/) running locally

**Recommended Hardware:**
- **M4 Max with 36GB+ unified memory** for concurrent generation + enhancement
- M1/M2/M3 with 16GB+ works but may require `--enhance-after` for large batches
- Generation speed: ~2-4 images/minute (z-image-turbo, 864x1152)

**Python Dependencies:**
- `openai` - LM Studio API client
- `click` - CLI framework
- `tracery` - Grammar expansion
- `mflux` - Image generation (optional, Apple Silicon only)
- `fastapi` + `sse-starlette` - Web UI server

## Installation

```bash
git clone https://github.com/fabian20ro/image-prompt-expander.git
cd image-prompt-expander

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Important:** Activate the virtual environment each new terminal session:
```bash
source venv/bin/activate
```

Then install [LM Studio](https://lmstudio.ai/), download a model (e.g., Qwen 2.5 7B, Llama 3.1 8B), and start the local server.

## Usage

### Web UI (Recommended)

Start the interactive web interface:

```bash
python src/cli.py --serve
```

This opens `http://localhost:8000` with:
- **New Generation Form**: Enter prompts and configure all generation parameters
- **Gallery Browser**: View and manage all existing galleries
- **Live Progress**: Real-time updates via SSE as images generate
- **Queue Management**: Queue multiple operations, kill running tasks

Gallery pages include:
- **Edit Grammar**: Modify Tracery grammar and regenerate prompts
- **Generate Images**: Queue individual or all images for generation
- **Enhance Images**: Apply SeedVR2 enhancement to individual or all images
- **Save to Archive**: Manually backup the current gallery state
- **Kill/Clear**: Stop current task or clear pending queue
- **Back to Index**: Navigate back to the master index

**Auto-Backup**: The system automatically creates backups before destructive operations (regenerating prompts when images exist, enhancing all images). Archives are saved as flat PNG files in `generated/saved/` with metadata embedded in PNG text chunks (prompt, model, settings). Archives appear as image grids in the "Archived Images" section on the index.

### CLI: Basic (Text Prompts Only)

```bash
# Generate 500 prompt variations
python src/cli.py -p "a dragon flying over mountains"

# Generate fewer variations
python src/cli.py -p "a cat sleeping on a bookshelf" -n 50

# Preview grammar without creating files
python src/cli.py -p "a cyberpunk city at night" --dry-run
```

### With Image Generation

```bash
# Generate prompts AND images (Apple Silicon only)
python src/cli.py -p "a dragon flying over mountains" -n 5 \
    --generate-images \
    --prefix dragon

# Multiple images per prompt
python src/cli.py -p "a mystical forest" -n 10 \
    --generate-images \
    --images-per-prompt 3 \
    --prefix forest

# Limit how many prompts get rendered
python src/cli.py -p "abstract art" -n 100 \
    --generate-images \
    --max-prompts 10 \
    --prefix abstract
```

### Custom Image Settings

```bash
# Different model (auto-selects optimized prompt structure)
python src/cli.py -p "portrait of a wizard" -n 5 -i \
    --model flux2-klein-4b \
    --prefix wizard

# Custom resolution and steps
python src/cli.py -p "landscape painting" -n 5 -i \
    --width 1024 --height 768 --steps 8 \
    --prefix landscape

# Reproducible with seed
python src/cli.py -p "abstract pattern" -n 3 -i \
    --seed 42 \
    --prefix pattern
```

### Image Enhancement (SeedVR2)

Enhance generated images with 2x upscaling using SeedVR2. Enhanced images replace the originals:

```bash
# Generate images with automatic 2x enhancement
python src/cli.py -p "a cat sleeping" -n 3 -i \
    --enhance \
    --prefix cat

# Adjust enhancement softness (0.0-1.0, default: 0.5)
python src/cli.py -p "portrait" -n 1 -i \
    --enhance --enhance-softness 0.3 \
    --prefix portrait

# Memory-efficient batch enhancement (for large batches)
# Defers enhancement until after all images are generated
python src/cli.py -p "a cat sleeping" -n 50 -i \
    --enhance --enhance-after \
    --prefix cat
```

### Standalone Enhancement

Enhance existing images in-place (replaces originals):

```bash
# Enhance a single image
python src/cli.py --enhance-images path/to/image.png

# Enhance all images in a folder
python src/cli.py --enhance-images generated/prompts/myrun/

# Enhance using glob pattern
python src/cli.py --enhance-images "generated/prompts/*/cat_*.png"

# With custom softness
python src/cli.py --enhance-images folder/ --enhance-softness 0.7
```

### Resume from Intermediate Steps

```bash
# Resume from cached grammar (skip LLM generation)
python src/cli.py --from-grammar generated/grammars/abc123.tracery.json \
    -n 100 --prefix dragon2

# Resume from existing prompts (generate images only)
python src/cli.py --from-prompts generated/prompts/abc123_20260124_122208 \
    --generate-images --images-per-prompt 2
```

### Cleanup

```bash
python src/cli.py --clean
```

## CLI Options

| Option | Description |
|--------|-------------|
| `-p, --prompt TEXT` | Image description to generate variations for |
| `-n, --count INT` | Number of variations (default: 500) |
| `-o, --output PATH` | Custom output directory |
| `--prefix TEXT` | Output file prefix (default: "image") |
| `--dry-run` | Preview grammar only |
| `--no-cache` | Force regenerate grammar |
| `--clean` | Remove all generated files |
| `--base-url TEXT` | LM Studio URL (default: http://localhost:1234/v1) |
| `--temperature FLOAT` | LLM temperature (default: 0.7) |
| `--from-grammar PATH` | Resume from existing grammar file (skip LLM generation) |
| `--from-prompts PATH` | Resume from existing prompts directory (images only) |
| `-i, --generate-images` | Enable mflux image generation |
| `--images-per-prompt INT` | Images per prompt (default: 1) |
| `--max-prompts INT` | Limit prompts to render |
| `-m, --model` | `z-image-turbo`, `flux2-klein-4b`, `flux2-klein-9b` |
| `--steps INT` | Inference steps |
| `--width INT` | Image width (default: 864) |
| `--height INT` | Image height (default: 1152) |
| `-q, --quantize` | Quantization: 3, 4, 5, 6, or 8 |
| `--seed INT` | Random seed |
| `--enhance` | Enable SeedVR2 2x enhancement (replaces original) |
| `--enhance-softness FLOAT` | Enhancement softness (0.0-1.0, default: 0.5) |
| `--enhance-after` | Defer enhancement to after all images generated (saves memory) |
| `--enhance-images PATH` | Enhance existing images in-place (file, folder, or glob) |
| `--resume` | Skip already-generated images when resuming interrupted runs |
| `--no-tiled-vae` | Disable tiled VAE decoding (more memory, faster) |
| `--serve` | Start interactive web UI at http://localhost:8000 |
| `--port INT` | Port for web UI server (default: 8000) |

## Output Structure

```
generated/
├── index.html                # Master index linking all galleries
├── queue.json                # Task queue persistence for web UI
├── grammars/                 # Cached grammars (by prompt hash)
├── prompts/{timestamp}_{hash}/    # Active generation runs
│   ├── dragon_0.txt          # First prompt
│   ├── dragon_0_0.png        # First image (enhanced in-place if --enhance)
│   ├── dragon_0_1.png        # Second image (if --images-per-prompt 2)
│   ├── dragon_1.txt          # Second prompt
│   ├── dragon_1_0.png
│   ├── ...
│   ├── dragon_gallery.html   # Gallery generated dynamically via --serve
│   ├── dragon_grammar.json   # Tracery grammar used
│   └── dragon_metadata.json  # Generation settings
└── saved/                    # Flat archived images
    ├── dragon_20260126_143052_0_0.png  # {prefix}_{timestamp}_{promptIdx}_{imgIdx}.png
    ├── dragon_20260126_143052_1_0.png  # Metadata embedded in PNG text chunks
    └── ...
```

The master index at `generated/index.html` provides a unified entry point to browse all generation runs with thumbnails and metadata. Archives appear as image grids in the "Archived Images" section, grouped by prefix and timestamp. Archive metadata (prompt, model, settings) is embedded directly in PNG text chunks for self-contained files. Grammars are cached and reused for identical prompts.

## How It Works

1. **Grammar Generation** - Your prompt is sent to a local LLM (recommended: GLM-4.7-Flash via LM Studio) with model-specific instructions to create a Tracery grammar. The grammar locks elements you specified and varies everything else. Different models use different prompt structures (camera-first for z-image-turbo, prose-based for flux2-klein).

2. **Prompt Expansion** - The grammar is expanded N times, randomly selecting from options to create diverse but coherent prompts.

3. **Image Generation** (optional) - Each prompt is rendered using mflux on Apple Silicon.

4. **Image Enhancement** (optional) - Images are enhanced in-place with SeedVR2 2x upscaling, replacing the originals with higher quality versions.

## Supported Models

| Model | Parameters | Default Steps | Notes |
|-------|------------|---------------|-------|
| z-image-turbo | 6B | 9 | Fast, good quality (default) |
| flux2-klein-4b | 4B | 4 | Very fast, lighter |
| flux2-klein-9b | 9B | 4 | Best quality |

Pre-quantized 4-bit versions are used automatically when available.

## Prompt Tips

- **Be specific** about constants: "a RED dragon with GOLDEN eyes"
- **Describe scene structure**: "a warrior standing on a cliff overlooking a battlefield"
- **Suggest variation dimensions**: "a cat in various cozy indoor settings"
- **Use FLUX-friendly language**: lighting ("golden hour"), atmosphere ("epic", "serene")
- **Front-load important elements** (FLUX prioritizes earlier content)

## Configuration

Settings can be overridden via environment variables with `PROMPT_GEN_` prefix:

```bash
# Use different LM Studio instance
export PROMPT_GEN_LM_STUDIO_URL="http://192.168.1.100:1234/v1"

# Change default image dimensions
export PROMPT_GEN_DEFAULT_WIDTH=1024
export PROMPT_GEN_DEFAULT_HEIGHT=768
```

## Troubleshooting

**"Connection refused"** - Start LM Studio and ensure the server is running.

**"mflux is required"** - Run `pip install mflux` (requires Apple Silicon).

**"Invalid JSON grammar"** - Try `--no-cache` or use a different LLM model.

**Slow generation** - First run downloads model weights. Use `--steps 4` or `flux2-klein-4b` for speed.

**Out of memory** - Use `--enhance-after` for batch enhancement (single model at a time). Reduce resolution, use `flux2-klein-4b`, or use `--no-tiled-vae` to trade memory for speed.

## Credits

- [Fifty Shades Generator](https://github.com/lisawray/fiftyshades) by Lisa Wray - original inspiration
- [Tracery](https://github.com/galaxykate/tracery) by Kate Compton - grammar expansion library
- [mflux](https://github.com/filipstrand/mflux) by Filip Strand - MLX-based image generation for Apple Silicon, including pre-quantized model weights
- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) by Tongyi-MAI - default image model (6B parameters)
- [FLUX models](https://blackforestlabs.ai/) by Black Forest Labs - flux2-klein image models

## License

See [LICENSE](LICENSE) file.

## Roadmap

- [ ] LoRA support for custom styles (if requested)
