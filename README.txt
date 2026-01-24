================================================================================
                          IMAGE-PROMPT-EXPANDER
================================================================================

A procedural image prompt generator that creates varied, high-quality prompts
for FLUX.2 image models, with optional local image generation using mflux.

Pipeline:
  User prompt -> LLM generates Tracery grammar -> Tracery produces N prompts
                                               -> (optional) mflux generates images

================================================================================
                              REQUIREMENTS
================================================================================

System Requirements:
  - Python 3.10 or higher
  - macOS with Apple Silicon (M1/M2/M3/M4) for image generation
  - LM Studio running locally (for grammar generation)

Python Dependencies:
  - openai>=1.0.0      (LM Studio API client)
  - click>=8.0.0       (CLI framework)
  - pydantic>=2.0.0    (Data validation)
  - tracery>=0.1.1     (Grammar expansion)
  - mflux>=0.15.0      (Image generation - optional, Apple Silicon only)

================================================================================
                              INSTALLATION
================================================================================

1. Clone the repository:

   git clone https://github.com/YOUR_USERNAME/image-prompt-expander.git
   cd image-prompt-expander

2. Create and activate a virtual environment:

   python3 -m venv venv
   source venv/bin/activate

3. Install dependencies:

   pip install -r requirements.txt

4. Install LM Studio:

   Download from: https://lmstudio.ai/

   - Launch LM Studio
   - Download a model (e.g., Qwen 2.5 7B, Llama 3.1 8B, or similar)
   - Start the local server (default: http://localhost:1234)

================================================================================
                                 USAGE
================================================================================

Basic Usage (Text Prompts Only)
-------------------------------

Generate 500 prompt variations from a description:

  python src/cli.py -p "a dragon flying over mountains"

Generate fewer variations:

  python src/cli.py -p "a cat sleeping on a bookshelf" -n 50

Preview the generated grammar without creating files:

  python src/cli.py -p "a cyberpunk city at night" --dry-run


With Image Generation
---------------------

Generate prompts AND images using mflux (Apple Silicon only):

  python src/cli.py -p "a dragon flying over mountains" -n 5 \
      --generate-images \
      --prefix dragon

Generate multiple images per prompt:

  python src/cli.py -p "a mystical forest" -n 10 \
      --generate-images \
      --images-per-prompt 3 \
      --prefix forest

Limit how many prompts get rendered to images:

  python src/cli.py -p "abstract art" -n 100 \
      --generate-images \
      --max-prompts 10 \
      --prefix abstract


Custom Image Settings
---------------------

Use a different model:

  python src/cli.py -p "portrait of a wizard" -n 5 -i \
      --model flux2-klein-4b \
      --prefix wizard

Custom resolution and steps:

  python src/cli.py -p "landscape painting" -n 5 -i \
      --width 1024 \
      --height 768 \
      --steps 8 \
      --prefix landscape

Reproducible generation with seed:

  python src/cli.py -p "abstract pattern" -n 3 -i \
      --seed 42 \
      --prefix pattern


Cleaning Up
-----------

Remove all generated files:

  python src/cli.py --clean

================================================================================
                            CLI OPTIONS REFERENCE
================================================================================

Prompt Generation Options:
--------------------------
  -p, --prompt TEXT         Image description to generate variations for
  -n, --count INTEGER       Number of prompt variations (default: 500)
  -o, --output PATH         Custom output directory
  --prefix TEXT             Prefix for output files (default: "image")
  --dry-run                 Show grammar without generating files
  --no-cache                Force regenerate grammar (skip cache)
  --clean                   Remove all generated files

LLM Options:
------------
  --base-url TEXT           LM Studio API URL (default: http://localhost:1234/v1)
  --temperature FLOAT       LLM temperature (default: 0.7)

Image Generation Options:
-------------------------
  -i, --generate-images     Enable image generation using mflux
  --images-per-prompt INT   Images per prompt (default: 1)
  --max-prompts INT         Limit prompts to render (default: all)

  -m, --model CHOICE        Model to use:
                              z-image-turbo   (default, 9 steps)
                              flux2-klein-4b  (4 steps)
                              flux2-klein-9b  (4 steps)

  --steps INTEGER           Inference steps (default: model-specific)
  --width INTEGER           Image width (default: 864)
  --height INTEGER          Image height (default: 1152)
  -q, --quantize CHOICE     Quantization: 3, 4, 5, 6, or 8 (default: 8)
  --seed INTEGER            Random seed for reproducibility

================================================================================
                             OUTPUT STRUCTURE
================================================================================

Text prompts only:

  generated/prompts/{hash}_{timestamp}/
  +-- image_0.txt              # First generated prompt
  +-- image_1.txt              # Second generated prompt
  +-- ...
  +-- image_499.txt            # 500th generated prompt
  +-- image_grammar.json       # Tracery grammar used
  +-- image_metadata.json      # Generation settings

With image generation (--prefix dragon):

  generated/prompts/{hash}_{timestamp}/
  +-- dragon_0.txt             # First prompt text
  +-- dragon_0_0.png           # First image from first prompt
  +-- dragon_0_1.png           # Second image from first prompt (if --images-per-prompt 2)
  +-- dragon_1.txt             # Second prompt text
  +-- dragon_1_0.png           # First image from second prompt
  +-- dragon_1_1.png           # Second image from second prompt
  +-- ...
  +-- dragon_grammar.json      # Tracery grammar used
  +-- dragon_metadata.json     # Generation settings with image params

Cached grammars (reused across runs with same prompt):

  generated/grammars/
  +-- {hash}.tracery.json      # Cached grammar
  +-- {hash}.meta.json         # Cache metadata

================================================================================
                              HOW IT WORKS
================================================================================

1. GRAMMAR GENERATION

   Your prompt is sent to a local LLM (via LM Studio) with a specialized
   system prompt that instructs it to create a Tracery grammar. The grammar
   defines rules for generating variations while preserving key elements
   you specified.

   Example: "a red dragon over mountains" ->
   - "red dragon" stays constant (you specified it)
   - Mountain types vary: "snow-capped peaks", "volcanic ridges", etc.
   - Lighting varies: "golden sunset", "misty dawn", etc.
   - Atmosphere varies: "epic and grand", "mysterious", etc.

2. PROMPT EXPANSION

   The Tracery grammar is expanded N times to produce unique prompt
   variations. Each expansion randomly selects from the defined options,
   creating diverse but coherent prompts.

3. IMAGE GENERATION (Optional)

   If --generate-images is enabled, each text prompt is fed to mflux
   (MLX-based FLUX implementation for Apple Silicon) to generate images
   locally on your Mac.

================================================================================
                           SUPPORTED MODELS
================================================================================

Model              Parameters   Default Steps   Notes
------------------ ------------ --------------- --------------------------------
z-image-turbo      6B           9               Fast, good quality (default)
flux2-klein-4b     4B           4               Very fast, lighter model
flux2-klein-9b     9B           4               Best quality, more VRAM needed

Pre-quantized 4-bit versions are used automatically when available:
  - filipstrand/Z-Image-Turbo-mflux-4bit
  - filipstrand/flux2-klein-4b-mflux-4bit

================================================================================
                         PROMPT ENGINEERING TIPS
================================================================================

The LLM generates better grammars when you:

1. BE SPECIFIC about what should stay constant:
   Good:  "a RED dragon with GOLDEN eyes"
   Less:  "a dragon" (more variation, less control)

2. DESCRIBE the scene structure:
   Good:  "a warrior standing on a cliff overlooking a battlefield"
   Less:  "a warrior" (lacks context for scene building)

3. SUGGEST variation dimensions:
   Good:  "a cat in various cozy indoor settings"
   Less:  "a cat" (grammar might vary unexpected things)

4. USE FLUX-FRIENDLY language:
   - Describe lighting: "golden hour", "dramatic shadows", "soft diffused light"
   - Describe atmosphere: "epic", "serene", "mysterious", "ethereal"
   - Front-load important elements (FLUX prioritizes earlier content)

================================================================================
                              EXAMPLES
================================================================================

Example 1: Character Portraits
------------------------------
python src/cli.py \
    -p "portrait of an elderly wizard with a long white beard, magical lighting" \
    -n 20 \
    --generate-images \
    --images-per-prompt 2 \
    --prefix wizard \
    --steps 6

Example 2: Landscapes
---------------------
python src/cli.py \
    -p "sweeping mountain landscape at various times of day" \
    -n 50 \
    --generate-images \
    --max-prompts 10 \
    --width 1152 \
    --height 864 \
    --prefix mountains

Example 3: Abstract Art
-----------------------
python src/cli.py \
    -p "abstract geometric patterns with vibrant colors" \
    -n 100 \
    --generate-images \
    --max-prompts 20 \
    --seed 12345 \
    --prefix abstract

Example 4: Product Photography
------------------------------
python src/cli.py \
    -p "minimalist product photo of a ceramic vase on a pedestal, studio lighting" \
    -n 30 \
    --generate-images \
    --prefix vase \
    --model flux2-klein-4b

================================================================================
                            TROUBLESHOOTING
================================================================================

"Error generating grammar: Connection refused"
----------------------------------------------
LM Studio is not running or not serving on the expected port.
Solution: Start LM Studio and ensure the local server is running.

"mflux is required for image generation"
----------------------------------------
The mflux package is not installed or you're not on Apple Silicon.
Solution: pip install mflux (requires macOS with M1/M2/M3/M4)

"Invalid JSON grammar"
----------------------
The LLM produced malformed JSON.
Solution: Try again with --no-cache, or try a different/larger LLM model.

Slow image generation
---------------------
First run downloads model weights (~5-10GB). Subsequent runs are faster.
Use --steps 4 for faster (but lower quality) results.
Use flux2-klein-4b for the fastest generation.

Out of memory
-------------
Reduce image resolution: --width 512 --height 512
Use a smaller model: --model flux2-klein-4b
Close other applications using GPU memory.

================================================================================
                               LICENSE
================================================================================

See LICENSE file for details.

================================================================================
                              CREDITS
================================================================================

- Fifty Shades Generator: Lisa Wray (https://github.com/lisawray/fiftyshades) -
  original inspiration for using procedural grammars to generate text
- Tracery: Kate Compton (https://github.com/galaxykate/tracery)
- mflux: Filip Strand (https://github.com/filipstrand/mflux)
- FLUX models: Black Forest Labs

================================================================================
