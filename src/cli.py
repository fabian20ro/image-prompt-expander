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
from image_generator import generate_image, clear_model_cache, SUPPORTED_MODELS, MODEL_DEFAULTS
from image_enhancer import enhance_image, collect_images
from gallery import create_gallery, update_gallery, generate_gallery_for_directory
from gallery_index import generate_master_index


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
@click.option(
    '--enhance',
    is_flag=True,
    help='Enable SeedVR2 2x enhancement after image generation'
)
@click.option(
    '--enhance-softness',
    default=0.5,
    type=float,
    help='Enhancement softness (0.0-1.0, default: 0.5)'
)
@click.option(
    '--enhance-after',
    is_flag=True,
    help='Defer enhancement to after all images are generated (saves memory)'
)
@click.option(
    '--enhance-images',
    type=str,
    help='Standalone: enhance existing images from file, folder, or glob pattern'
)
@click.option(
    '--resume',
    is_flag=True,
    help='Skip already-generated images when resuming interrupted runs'
)
@click.option(
    '--gallery',
    type=click.Path(exists=True, path_type=Path),
    help='Standalone: generate gallery.html for existing prompts directory'
)
@click.option(
    '--no-tiled-vae',
    is_flag=True,
    help='Disable tiled VAE decoding (uses more memory but may be faster)'
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
    enhance: bool,
    enhance_softness: float,
    enhance_after: bool,
    enhance_images: str | None,
    resume: bool,
    gallery: Path | None,
    no_tiled_vae: bool,
):
    """
    Generate FLUX.2 Klein image prompt variations using LLM-powered Tracery grammars.

    Example:
        python cli.py -p "a dragon flying over mountains" -n 500
        python cli.py --clean  # Remove all generated files

    With image generation:
        python cli.py -p "a dragon flying over mountains" -n 5 \\
            --generate-images --images-per-prompt 3 --prefix dragon

    With image generation + SeedVR2 2x enhancement:
        python cli.py -p "a cat" -n 1 --generate-images --enhance --prefix test

    Resume from existing grammar:
        python cli.py --from-grammar generated/grammars/abc123.tracery.json -n 100

    Resume from existing prompts (images only):
        python cli.py --from-prompts generated/prompts/abc123_20260124_122208 \\
            --generate-images --images-per-prompt 2

    Standalone enhancement (no image generation):
        python cli.py --enhance-images path/to/image.png
        python cli.py --enhance-images path/to/folder/
        python cli.py --enhance-images "generated/prompts/*/test_*.png"
    """
    # Handle --clean
    if clean:
        removed = clean_generated()
        click.echo(f"Cleaned {removed} items from generated/")
        if not prompt and not from_grammar and not from_prompts and not enhance_images and not gallery:
            return

    # STANDALONE MODE: Enhance existing images
    if enhance_images:
        click.echo(f"Collecting images from: {enhance_images}")
        try:
            images = collect_images(enhance_images)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        click.echo(f"Enhancing {len(images)} images with SeedVR2 (2x upscale, softness={enhance_softness})...")

        # Set defaults for quantize if not specified
        if quantize is None:
            quantize = 8

        enhanced_count = 0
        for idx, img_path in enumerate(images):
            enhanced_count += 1
            click.echo(f"  [{enhanced_count}/{len(images)}] Enhancing {img_path.name} (replacing original)...")

            try:
                enhance_image(
                    image_path=img_path,
                    output_path=img_path,
                    softness=enhance_softness,
                    seed=seed,
                    quantize=quantize,
                    tiled_vae=not no_tiled_vae,
                )
            except ImportError as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)
            except Exception as e:
                click.echo(f"Error enhancing image: {e}", err=True)
                sys.exit(1)

            # Increment seed for next image (if seed was specified)
            if seed is not None:
                seed += 1

        click.echo(f"\nEnhanced {enhanced_count} images")
        return

    # STANDALONE MODE: Generate gallery for existing prompts directory
    if gallery:
        click.echo(f"Generating gallery for: {gallery}")
        try:
            gallery_path = generate_gallery_for_directory(gallery)
            gallery_url = f"file://{gallery_path.resolve()}"
            click.echo(f"Gallery created: {gallery_url}")

            # Update master index
            index_path = generate_master_index(GENERATED_DIR)
            index_url = f"file://{index_path.resolve()}"
            click.echo(f"Master index: {index_url}")
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        return

    # Validation: mutual exclusivity
    if from_grammar and from_prompts:
        click.echo("Error: Cannot use both --from-grammar and --from-prompts", err=True)
        sys.exit(1)

    if from_prompts and not generate_images:
        click.echo("Error: --from-prompts requires --generate-images", err=True)
        sys.exit(1)

    if not prompt and not from_grammar and not from_prompts and not clean and not enhance_images and not gallery:
        click.echo("Error: --prompt is required (or use --from-grammar/--from-prompts/--enhance-images/--gallery)", err=True)
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
        user_prompt = existing_metadata.get("user_prompt", "unknown")

        # Load grammar if available
        grammar = None
        grammar_file = prompts_dir / f"{detected_prefix}_grammar.json"
        if grammar_file.exists():
            grammar = grammar_file.read_text()

        # Check for raw response file
        raw_response_file = None
        raw_file = prompts_dir / f"{detected_prefix}_raw_response.txt"
        if raw_file.exists():
            raw_response_file = f"{detected_prefix}_raw_response.txt"

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
            "enhance": enhance,
            "enhance_softness": enhance_softness if enhance else None,
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
        meta_path = from_grammar.with_suffix('.metaprompt.json')
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            user_prompt = meta.get("user_prompt", "unknown")
        else:
            user_prompt = "unknown"

        # Try to find raw response in same directory (e.g., abc123.raw.txt)
        grammar_hash = from_grammar.stem.replace('.tracery', '')
        raw_path = from_grammar.parent / f"{grammar_hash}.raw.txt"
        raw_response = raw_path.read_text() if raw_path.exists() else None

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
            output = GENERATED_DIR / "prompts" / f"{timestamp}_{grammar_stem}"

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
                "enhance": enhance,
                "enhance_softness": enhance_softness if enhance else None,
            }

        # Save metadata and grammar with prefix
        metadata_file = output / f"{prefix}_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

        grammar_file = output / f"{prefix}_grammar.json"
        grammar_file.write_text(grammar)

        # Save raw LLM response if available
        raw_response_file = None
        if raw_response:
            raw_file = output / f"{prefix}_raw_response.txt"
            raw_file.write_text(raw_response)
            raw_response_file = f"{prefix}_raw_response.txt"

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
            grammar, was_cached, raw_response = generate_grammar(
                user_prompt=prompt,
                base_url=base_url,
                use_cache=not no_cache,
                temperature=temperature,
                model=model,
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
            output = GENERATED_DIR / "prompts" / f"{timestamp}_{prompt_hash}"

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
                "enhance": enhance,
                "enhance_softness": enhance_softness if enhance else None,
            }

        # Save metadata and grammar with prefix
        metadata_file = output / f"{prefix}_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

        grammar_file = output / f"{prefix}_grammar.json"
        grammar_file.write_text(grammar)

        # Save raw LLM response if available
        raw_response_file = None
        if raw_response:
            raw_file = output / f"{prefix}_raw_response.txt"
            raw_file.write_text(raw_response)
            raw_response_file = f"{prefix}_raw_response.txt"

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

        # Create gallery before starting image generation
        gallery_path = create_gallery(
            output, prefix, prompts_to_render, images_per_prompt,
            grammar=grammar, raw_response_file=raw_response_file
        )
        gallery_url = f"file://{gallery_path.resolve()}"
        click.echo(f"Gallery: {gallery_url}")

        # Update master index
        index_path = generate_master_index(GENERATED_DIR)
        index_url = f"file://{index_path.resolve()}"
        click.echo(f"Master index: {index_url}")

        click.echo(f"\nGenerating {total_images} images ({len(prompts_to_render)} prompts x {images_per_prompt} images each)...")
        click.echo(f"Model: {model}, Steps: {steps or MODEL_DEFAULTS[model]['steps']}, Size: {width}x{height}")

        current_seed = seed
        generated_count = 0
        skipped_count = 0
        images_to_enhance = []  # Track images for batch enhancement

        for prompt_idx, prompt_text in enumerate(prompts_to_render):
            for image_idx in range(images_per_prompt):
                output_path = output / f"{prefix}_{prompt_idx}_{image_idx}.png"

                # Resume logic: skip existing images
                if resume and output_path.exists():
                    skipped_count += 1
                    generated_count += 1  # Still count for progress display
                    click.echo(f"  [{generated_count}/{total_images}] Skipping {output_path.name} (exists)")
                    # Increment seed even for skipped images to maintain reproducibility
                    if current_seed is not None:
                        current_seed += 1
                    continue

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
                        tiled_vae=not no_tiled_vae,
                    )
                except ImportError as e:
                    click.echo(f"Error: {e}", err=True)
                    sys.exit(1)
                except Exception as e:
                    click.echo(f"Error generating image: {e}", err=True)
                    sys.exit(1)

                # Update gallery after each image
                update_gallery(gallery_path, output_path, prompt_text, generated_count - skipped_count, total_images - skipped_count)

                # Enhance the image if requested (replaces original)
                if enhance and not enhance_after:
                    # Interleaved mode: enhance immediately after generation
                    click.echo(f"    Enhancing {output_path.name}...")
                    try:
                        enhance_image(
                            image_path=output_path,
                            output_path=output_path,
                            softness=enhance_softness,
                            seed=current_seed,
                            quantize=quantize,
                            tiled_vae=not no_tiled_vae,
                        )
                    except ImportError as e:
                        click.echo(f"Error: {e}", err=True)
                        sys.exit(1)
                    except Exception as e:
                        click.echo(f"Error enhancing image: {e}", err=True)
                        sys.exit(1)
                elif enhance and enhance_after:
                    # Batch mode: track for later enhancement
                    images_to_enhance.append((output_path, current_seed))

                # Increment seed for next image (if seed was specified)
                if current_seed is not None:
                    current_seed += 1

        actual_generated = generated_count - skipped_count
        if skipped_count > 0:
            click.echo(f"\nGenerated {actual_generated} images, skipped {skipped_count} existing images in: {output}")
        else:
            click.echo(f"\nGenerated {actual_generated} images in: {output}")

        # Batch enhancement phase (when --enhance-after is used)
        if enhance and enhance_after and images_to_enhance:
            click.echo(f"\nClearing image generator from memory...")
            clear_model_cache()

            click.echo(f"Enhancing {len(images_to_enhance)} images with SeedVR2...")
            for idx, (image_path, image_seed) in enumerate(images_to_enhance, 1):
                click.echo(f"  [{idx}/{len(images_to_enhance)}] Enhancing {image_path.name}...")
                try:
                    enhance_image(
                        image_path=image_path,
                        output_path=image_path,
                        softness=enhance_softness,
                        seed=image_seed,
                        quantize=quantize,
                        tiled_vae=not no_tiled_vae,
                    )
                except ImportError as e:
                    click.echo(f"Error: {e}", err=True)
                    sys.exit(1)
                except Exception as e:
                    click.echo(f"Error enhancing image: {e}", err=True)
                    sys.exit(1)
            click.echo(f"Enhanced {len(images_to_enhance)} images")


if __name__ == '__main__':
    main()
