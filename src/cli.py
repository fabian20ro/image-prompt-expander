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
    default='image',
    help='Prefix for output files (default: "image")'
)
@click.option(
    '--model', '-m',
    default='z-image-turbo',
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
    default=864,
    type=int,
    help='Image width in pixels (default: 864)'
)
@click.option(
    '--height',
    default=1152,
    type=int,
    help='Image height in pixels (default: 1152)'
)
@click.option(
    '--quantize', '-q',
    default=8,
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
    prefix: str,
    model: str,
    steps: int | None,
    width: int,
    height: int,
    quantize: int,
    max_prompts: int | None,
    seed: int | None,
):
    """
    Generate FLUX.2 Klein image prompt variations using LLM-powered Tracery grammars.

    Example:
        python cli.py -p "a dragon flying over mountains" -n 500
        python cli.py --clean  # Remove all generated files

    With image generation:
        python cli.py -p "a dragon flying over mountains" -n 5 \\
            --generate-images --images-per-prompt 3 --prefix dragon
    """
    # Handle --clean
    if clean:
        removed = clean_generated()
        click.echo(f"Cleaned {removed} items from generated/")
        if not prompt:
            return

    # Require prompt for generation
    if not prompt:
        click.echo("Error: --prompt is required (unless using --clean)", err=True)
        sys.exit(1)

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
        output = Path(__file__).parent.parent / "generated" / "prompts" / f"{prompt_hash}_{timestamp}"

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

    # Generate images if requested
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
