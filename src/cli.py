#!/usr/bin/env python3
"""CLI entry point for the FLUX.2 Klein image prompt generator."""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import click

from grammar_generator import generate_grammar, hash_prompt
from tracery_runner import run_tracery, TraceryError
from image_generator import generate_image, SUPPORTED_MODELS, MODEL_DEFAULTS


GENERATED_DIR = Path(__file__).parent.parent / "generated"


def clean_generated():
    """Remove all generated files (grammars and prompts)."""
    grammars_dir = GENERATED_DIR / "grammars"
    prompts_dir = GENERATED_DIR / "prompts"

    count = 0
    for d in [grammars_dir, prompts_dir]:
        if d.exists():
            for item in d.iterdir():
                if item.is_file():
                    item.unlink()
                    count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    count += 1
    return count


@click.command()
@click.option(
    '-p', '--prompt',
    default=None,
    help='Image description to generate variations for'
)
@click.option(
    '--clean',
    is_flag=True,
    help='Remove all generated files (grammars and prompts)'
)
@click.option(
    '-n', '--count',
    default=500,
    type=int,
    help='Number of variations to generate (default: 500)'
)
@click.option(
    '-o', '--output',
    type=click.Path(path_type=Path),
    help='Output directory (default: generated/prompts/{hash}_{timestamp}/)'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Generate and display grammar without running Dada Engine'
)
@click.option(
    '--no-cache',
    is_flag=True,
    help='Skip grammar cache and regenerate'
)
@click.option(
    '--base-url',
    default='http://localhost:1234/v1',
    help='LM Studio API base URL'
)
@click.option(
    '--temperature',
    default=0.7,
    type=float,
    help='LLM temperature for grammar generation (default: 0.7)'
)
@click.option(
    '--generate-images', '-i',
    is_flag=True,
    help='Generate images for each prompt using mflux'
)
@click.option(
    '--images-per-prompt',
    default=1,
    type=int,
    help='Number of images to generate per prompt (default: 1)'
)
@click.option(
    '--prefix',
    default=None,
    help='Prefix for output files (default: "image")'
)
@click.option(
    '--model', '-m',
    default=None,
    type=click.Choice(SUPPORTED_MODELS),
    help='mflux model to use (default: z-image-turbo)'
)
@click.option(
    '--steps',
    type=int,
    default=None,
    help='Inference steps (default: model-specific)'
)
@click.option(
    '--width',
    default=None,
    type=int,
    help='Image width in pixels (default: 864)'
)
@click.option(
    '--height',
    default=None,
    type=int,
    help='Image height in pixels (default: 1152)'
)
@click.option(
    '--quantize', '-q',
    default=None,
    type=click.Choice([3, 4, 5, 6, 8], case_sensitive=False),
    help='Quantization level for model (default: 8)'
)
@click.option(
    '--max-prompts',
    type=int,
    default=None,
    help='Limit number of prompts to render images for (default: all)'
)
@click.option(
    '--seed',
    type=int,
    default=None,
    help='Random seed for image generation (default: random)'
)
@click.option(
    '--from-grammar',
    type=click.Path(exists=True, path_type=Path),
    help='Resume from existing grammar file (skip LLM generation)'
)
@click.option(
    '--from-prompts',
    type=click.Path(exists=True, path_type=Path),
    help='Resume from existing prompts directory (images only)'
)
def main(
    prompt: str | None,
    clean: bool,
    count: int,
    output: Path | None,
    dry_run: bool,
    no_cache: bool,
    base_url: str,
    temperature: float,
    generate_images: bool,
    images_per_prompt: int,
    prefix: str | None,
    model: str | None,
    steps: int | None,
    width: int | None,
    height: int | None,
    quantize: int | None,
    max_prompts: int | None,
    seed: int | None,
    from_grammar: Path | None,
    from_prompts: Path | None,
):
    """
    Generate FLUX.2 Klein image prompt variations using LLM-powered Tracery grammars.

    Example:
        python cli.py -p "a dragon flying over mountains" -n 500
        python cli.py --clean  # Remove all generated files

    With image generation:
        python cli.py -p "a dragon flying over mountains" -n 5 \\
            --generate-images --images-per-prompt 3 --prefix dragon

    Resume from existing grammar:
        python cli.py --from-grammar generated/grammars/abc123.tracery.json -n 100

    Resume from existing prompts (images only):
        python cli.py --from-prompts generated/prompts/abc123_20260124_122208 \\
            --generate-images --images-per-prompt 2
    """
    # Handle --clean
    if clean:
        removed = clean_generated()
        click.echo(f"Cleaned {removed} items from generated/")
        if not prompt and not from_grammar and not from_prompts:
            return

    # Validation: mutual exclusivity
    if from_grammar and from_prompts:
        click.echo("Error: Cannot use both --from-grammar and --from-prompts", err=True)
        sys.exit(1)

    if from_prompts and not generate_images:
        click.echo("Error: --from-prompts requires --generate-images", err=True)
        sys.exit(1)

    if not prompt and not from_grammar and not from_prompts and not clean:
        click.echo("Error: --prompt is required (or use --from-grammar/--from-prompts)", err=True)
        sys.exit(1)

    # Warn if --prompt is provided with --from-grammar or --from-prompts
    if prompt and (from_grammar or from_prompts):
        click.echo("Warning: --prompt is ignored when using --from-grammar or --from-prompts", err=True)

    # Set defaults for optional parameters
    if prefix is None:
        prefix = "image"
    if model is None:
        model = "z-image-turbo"
    if width is None:
        width = 864
    if height is None:
        height = 1152
    if quantize is None:
        quantize = 8

    # MODE C: From prompts (images only)
    if from_prompts:
        prompts_dir = from_prompts

        # Find metadata file
        meta_files = list(prompts_dir.glob("*_metadata.json"))
        if not meta_files:
            click.echo(f"Error: No metadata file found in {prompts_dir}", err=True)
            sys.exit(1)

        existing_metadata = json.loads(meta_files[0].read_text())
        detected_prefix = existing_metadata.get("prefix", "image")

        # Load existing prompts (files with exactly one underscore in stem, excluding metadata/grammar)
        prompt_files = sorted(prompts_dir.glob(f"{detected_prefix}_*.txt"))
        prompt_files = [f for f in prompt_files if f.stem.count('_') == 1]

        if not prompt_files:
            click.echo(f"Error: No prompt files found in {prompts_dir}", err=True)
            sys.exit(1)

        outputs = [f.read_text() for f in prompt_files]
        click.echo(f"Loaded {len(outputs)} prompts from {prompts_dir}")

        # Use existing metadata as defaults for image settings (CLI overrides)
        existing_image_settings = existing_metadata.get("image_generation", {})

        # Generate images in same directory
        output = prompts_dir
        was_cached = None
        grammar = None
        user_prompt = existing_metadata.get("user_prompt", "unknown")

        # Prepare metadata update
        metadata = existing_metadata.copy()
        metadata["image_generation"] = {
            "enabled": True,
            "model": model,
            "steps": steps if steps is not None else MODEL_DEFAULTS[model]["steps"],
            "width": width,
            "height": height,
            "quantize": quantize,
            "images_per_prompt": images_per_prompt,
            "max_prompts": max_prompts,
            "seed": seed,
            "resumed_from": str(prompts_dir),
            "original_settings": existing_image_settings if existing_image_settings else None,
        }

        # Update metadata file
        metadata_file = output / f"{detected_prefix}_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

        # Use detected prefix for output files
        prefix = detected_prefix

    # MODE B: From grammar (skip LLM generation)
    elif from_grammar:
        click.echo(f"Loading grammar from: {from_grammar}")
        grammar = from_grammar.read_text()

        # Try to find metadata for original prompt
        meta_path = from_grammar.with_suffix('.meta.json')
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            user_prompt = meta.get("user_prompt", "unknown")
        else:
            user_prompt = "unknown"

        was_cached = True  # Treat as cached since we loaded from file

        # Dry run: just show the grammar
        if dry_run:
            click.echo("\n--- Loaded Grammar ---\n")
            click.echo(grammar)
            click.echo("\n--- End Grammar ---")
            return

        # Create new output directory using grammar filename
        grammar_stem = from_grammar.stem.replace('.tracery', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output is None:
            output = GENERATED_DIR / "prompts" / f"{grammar_stem}_{timestamp}"

        output.mkdir(parents=True, exist_ok=True)

        # Run Tracery
        click.echo(f"Generating {count} variations...")

        try:
            outputs = run_tracery(grammar, count=count)
        except TraceryError as e:
            click.echo(f"Error running Tracery: {e}", err=True)
            sys.exit(1)

        # Save outputs with prefix naming
        for i, text in enumerate(outputs):
            output_file = output / f"{prefix}_{i}.txt"
            output_file.write_text(text)

        # Build metadata
        metadata = {
            "source": "from_grammar",
            "grammar_path": str(from_grammar),
            "user_prompt": user_prompt,
            "count": len(outputs),
            "created_at": datetime.now().isoformat(),
            "grammar_cached": True,
            "prefix": prefix,
        }

        # Add image generation metadata if enabled
        if generate_images:
            effective_steps = steps if steps is not None else MODEL_DEFAULTS[model]["steps"]
            metadata["image_generation"] = {
                "enabled": True,
                "model": model,
                "steps": effective_steps,
                "width": width,
                "height": height,
                "quantize": quantize,
                "images_per_prompt": images_per_prompt,
                "max_prompts": max_prompts,
                "seed": seed,
            }

        # Save metadata and grammar with prefix
        metadata_file = output / f"{prefix}_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

        grammar_file = output / f"{prefix}_grammar.json"
        grammar_file.write_text(grammar)

        click.echo(f"Generated {len(outputs)} prompts in: {output}")

        # Show a sample
        if outputs:
            click.echo("\n--- Sample Output ---")
            click.echo(outputs[0])
            click.echo("--- End Sample ---")

    # MODE A: Full pipeline (default)
    else:
        click.echo(f"Generating grammar for: {prompt}")

        # Generate grammar via LM Studio
        try:
            grammar, was_cached = generate_grammar(
                user_prompt=prompt,
                base_url=base_url,
                use_cache=not no_cache,
                temperature=temperature
            )
        except Exception as e:
            click.echo(f"Error generating grammar: {e}", err=True)
            click.echo("Make sure LM Studio is running at " + base_url, err=True)
            sys.exit(1)

        if was_cached:
            click.echo("Using cached grammar")
        else:
            click.echo("Generated new grammar via LM Studio")

        # Dry run: just show the grammar
        if dry_run:
            click.echo("\n--- Generated Grammar ---\n")
            click.echo(grammar)
            click.echo("\n--- End Grammar ---")
            return

        # Determine output directory
        if output is None:
            prompt_hash = hash_prompt(prompt)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = GENERATED_DIR / "prompts" / f"{prompt_hash}_{timestamp}"

        output.mkdir(parents=True, exist_ok=True)

        # Run Tracery
        click.echo(f"Generating {count} variations...")

        try:
            outputs = run_tracery(grammar, count=count)
        except TraceryError as e:
            click.echo(f"Error running Tracery: {e}", err=True)
            sys.exit(1)

        # Save outputs with prefix naming
        for i, text in enumerate(outputs):
            output_file = output / f"{prefix}_{i}.txt"
            output_file.write_text(text)

        # Build metadata
        metadata = {
            "user_prompt": prompt,
            "count": len(outputs),
            "created_at": datetime.now().isoformat(),
            "grammar_cached": was_cached,
            "prefix": prefix,
        }

        # Add image generation metadata if enabled
        if generate_images:
            effective_steps = steps if steps is not None else MODEL_DEFAULTS[model]["steps"]
            metadata["image_generation"] = {
                "enabled": True,
                "model": model,
                "steps": effective_steps,
                "width": width,
                "height": height,
                "quantize": quantize,
                "images_per_prompt": images_per_prompt,
                "max_prompts": max_prompts,
                "seed": seed,
            }

        # Save metadata and grammar with prefix
        metadata_file = output / f"{prefix}_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

        grammar_file = output / f"{prefix}_grammar.json"
        grammar_file.write_text(grammar)

        click.echo(f"Generated {len(outputs)} prompts in: {output}")

        # Show a sample
        if outputs:
            click.echo("\n--- Sample Output ---")
            click.echo(outputs[0])
            click.echo("--- End Sample ---")

    # Generate images if requested (shared by all modes)
    if generate_images:
        prompts_to_render = outputs[:max_prompts] if max_prompts else outputs
        total_images = len(prompts_to_render) * images_per_prompt

        click.echo(f"\nGenerating {total_images} images ({len(prompts_to_render)} prompts x {images_per_prompt} images each)...")
        click.echo(f"Model: {model}, Steps: {steps or MODEL_DEFAULTS[model]['steps']}, Size: {width}x{height}")

        current_seed = seed
        generated_count = 0

        for prompt_idx, prompt_text in enumerate(prompts_to_render):
            for image_idx in range(images_per_prompt):
                output_path = output / f"{prefix}_{prompt_idx}_{image_idx}.png"
                generated_count += 1

                click.echo(f"  [{generated_count}/{total_images}] Generating {output_path.name}...")

                try:
                    generate_image(
                        prompt=prompt_text,
                        output_path=output_path,
                        model=model,
                        seed=current_seed,
                        steps=steps,
                        width=width,
                        height=height,
                        quantize=quantize,
                    )
                except ImportError as e:
                    click.echo(f"Error: {e}", err=True)
                    sys.exit(1)
                except Exception as e:
                    click.echo(f"Error generating image: {e}", err=True)
                    sys.exit(1)

                # Increment seed for next image (if seed was specified)
                if current_seed is not None:
                    current_seed += 1

        click.echo(f"\nGenerated {generated_count} images in: {output}")


if __name__ == '__main__':
    main()
