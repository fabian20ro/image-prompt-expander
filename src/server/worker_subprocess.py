#!/usr/bin/env python3
"""Worker subprocess for executing generation tasks.

This script is spawned by the main server to handle heavy operations
in isolation. It reads task configuration from a JSON file, executes
the appropriate operation, and writes progress to stdout as JSON lines.

Usage:
    python worker_subprocess.py <task.json>
"""

import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add parent directories to path for imports
SRC_DIR = Path(__file__).parent.parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from grammar_generator import generate_grammar, hash_prompt
from tracery_runner import run_tracery, TraceryError
from image_generator import generate_image, clear_model_cache, MODEL_DEFAULTS
from image_enhancer import enhance_image
from gallery import create_gallery, update_gallery
from gallery_index import generate_master_index

GENERATED_DIR = ROOT_DIR / "generated"

# Global log file handle (set when we know the output directory)
_log_file = None


def set_log_file(log_path: Path) -> None:
    """Set the log file path for this task."""
    global _log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(log_path, 'a', buffering=1)  # Line-buffered
    _log_file.write(f"\n{'='*60}\n")
    _log_file.write(f"Task started at {datetime.now().isoformat()}\n")
    _log_file.write(f"{'='*60}\n")


def close_log_file() -> None:
    """Close the log file."""
    global _log_file
    if _log_file:
        _log_file.write(f"\n{'='*60}\n")
        _log_file.write(f"Task ended at {datetime.now().isoformat()}\n")
        _log_file.write(f"{'='*60}\n")
        _log_file.close()
        _log_file = None


def log_to_file(message: str) -> None:
    """Write a message to the log file if set."""
    if _log_file:
        timestamp = datetime.now().strftime("%H:%M:%S")
        _log_file.write(f"[{timestamp}] {message}\n")


def emit_progress(stage: str, current: int = 0, total: int = 0, message: str = ""):
    """Emit progress update to stdout."""
    data = {
        "type": "progress",
        "stage": stage,
        "current": current,
        "total": total,
        "message": message,
    }
    print(json.dumps(data), flush=True)
    if message:
        log_to_file(message)


def emit_result(success: bool, data: dict | None = None, error: str | None = None):
    """Emit final result to stdout."""
    result = {
        "type": "result",
        "success": success,
        "data": data,
        "error": error,
    }
    print(json.dumps(result), flush=True)
    if success:
        log_to_file(f"Task completed successfully: {data}")
    else:
        log_to_file(f"Task failed: {error}")


def emit_image_ready(run_id: str, path: str):
    """Emit notification that an image is ready."""
    data = {
        "type": "image_ready",
        "run_id": run_id,
        "path": path,
    }
    print(json.dumps(data), flush=True)


def sync_file(path: Path) -> None:
    """Ensure file is flushed to disk before notifying.

    This prevents race conditions where the SSE event arrives
    before the file is fully written to disk.
    """
    with open(path, 'r+b') as f:
        f.flush()
        os.fsync(f.fileno())


class Heartbeat:
    """Context manager for emitting periodic heartbeats during long operations."""

    def __init__(self, message: str = "Working...", interval: int = 30):
        """Initialize heartbeat.

        Args:
            message: Message to emit with heartbeat
            interval: Seconds between heartbeats
        """
        self.message = message
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None

    def _heartbeat_loop(self):
        """Emit heartbeat every interval seconds until stopped."""
        while not self._stop_event.wait(timeout=self.interval):
            emit_progress("heartbeat", 0, 0, self.message)

    def __enter__(self):
        """Start the heartbeat thread."""
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the heartbeat thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)
        return False


def run_generate_pipeline(params: dict):
    """Run full generation pipeline: LLM -> Tracery -> Images."""
    prompt = params["prompt"]
    count = params.get("count", 50)
    prefix = params.get("prefix", "image")
    model = params.get("model", "z-image-turbo")
    temperature = params.get("temperature", 0.7)
    no_cache = params.get("no_cache", False)
    generate_images = params.get("generate_images", False)
    images_per_prompt = params.get("images_per_prompt", 1)
    width = params.get("width", 864)
    height = params.get("height", 1152)
    steps = params.get("steps")
    quantize = params.get("quantize", 8)
    seed = params.get("seed")
    max_prompts = params.get("max_prompts")
    tiled_vae = params.get("tiled_vae", True)
    enhance = params.get("enhance", False)
    enhance_softness = params.get("enhance_softness", 0.5)
    enhance_after = params.get("enhance_after", False)

    # Stage 1: Generate grammar
    cache_msg = " (ignoring cache)" if no_cache else ""
    emit_progress("generating_grammar", 0, 1, f"Generating grammar{cache_msg} for: {prompt[:50]}...")

    try:
        grammar, was_cached, raw_response = generate_grammar(
            user_prompt=prompt,
            use_cache=not no_cache,
            temperature=temperature,
            model=model,
        )
    except Exception as e:
        emit_result(False, error=f"Grammar generation failed: {e}")
        return

    emit_progress("generating_grammar", 1, 1, "Grammar generated")

    # Stage 2: Run Tracery
    emit_progress("expanding_prompts", 0, count, f"Expanding {count} prompts...")

    try:
        outputs = run_tracery(grammar, count=count)
    except TraceryError as e:
        emit_result(False, error=f"Tracery expansion failed: {e}")
        return

    emit_progress("expanding_prompts", count, count, f"Generated {len(outputs)} prompts")

    # Create output directory
    prompt_hash = hash_prompt(prompt)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = GENERATED_DIR / "prompts" / f"{timestamp}_{prompt_hash}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = output_dir.name

    # Set up per-task log file
    set_log_file(output_dir / f"{prefix}_worker.log")
    log_to_file(f"Starting generation pipeline for: {prompt[:100]}...")

    # Save prompts
    for i, text in enumerate(outputs):
        output_file = output_dir / f"{prefix}_{i}.txt"
        output_file.write_text(text)

    # Build metadata
    metadata = {
        "user_prompt": prompt,
        "count": len(outputs),
        "created_at": datetime.now().isoformat(),
        "grammar_cached": was_cached,
        "prefix": prefix,
    }

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

    # Save metadata and grammar
    metadata_file = output_dir / f"{prefix}_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))

    grammar_file = output_dir / f"{prefix}_grammar.json"
    grammar_file.write_text(grammar)

    # Save raw LLM response
    raw_response_file = None
    if raw_response:
        raw_file = output_dir / f"{prefix}_raw_response.txt"
        raw_file.write_text(raw_response)
        raw_response_file = f"{prefix}_raw_response.txt"

    # Always create gallery (even without images)
    prompts_to_render = outputs[:max_prompts] if max_prompts else outputs
    gallery_path = create_gallery(
        output_dir, prefix, prompts_to_render, images_per_prompt if generate_images else 0,
        grammar=grammar, raw_response_file=raw_response_file
    )

    # Update master index
    generate_master_index(GENERATED_DIR)

    # Stage 3: Generate images (if requested)
    if generate_images:
        total_images = len(prompts_to_render) * images_per_prompt
        emit_progress("generating_images", 0, total_images, "Starting image generation...")

        current_seed = seed
        generated_count = 0
        images_to_enhance = []

        for prompt_idx, prompt_text in enumerate(prompts_to_render):
            for image_idx in range(images_per_prompt):
                output_path = output_dir / f"{prefix}_{prompt_idx}_{image_idx}.png"
                generated_count += 1

                emit_progress(
                    "generating_images",
                    generated_count,
                    total_images,
                    f"Generating {output_path.name}..."
                )

                try:
                    with Heartbeat(f"Generating {output_path.name}..."):
                        generate_image(
                            prompt=prompt_text,
                            output_path=output_path,
                            model=model,
                            seed=current_seed,
                            steps=steps,
                            width=width,
                            height=height,
                            quantize=quantize,
                            tiled_vae=tiled_vae,
                        )
                except Exception as e:
                    context = f"model={model}, prompt_idx={prompt_idx}, image_idx={image_idx}"
                    emit_progress("error_detail", 0, 0, traceback.format_exc())
                    emit_result(False, error=f"Image generation failed ({context}): {e}")
                    return

                # Sync file to disk before notifying
                sync_file(output_path)

                # Update gallery
                update_gallery(gallery_path, output_path, prompt_text, generated_count, total_images)

                # Notify about new image
                emit_image_ready(run_id, output_path.name)

                # Enhancement
                if enhance and not enhance_after:
                    emit_progress(
                        "enhancing_image",
                        generated_count,
                        total_images,
                        f"Enhancing {output_path.name}..."
                    )
                    try:
                        with Heartbeat(f"Enhancing {output_path.name}..."):
                            enhance_image(
                                image_path=output_path,
                                output_path=output_path,
                                softness=enhance_softness,
                                seed=current_seed,
                                quantize=quantize,
                                tiled_vae=tiled_vae,
                            )
                    except Exception as e:
                        context = f"prompt_idx={prompt_idx}, image_idx={image_idx}"
                        emit_progress("error_detail", 0, 0, traceback.format_exc())
                        emit_result(False, error=f"Enhancement failed ({context}): {e}")
                        return

                    sync_file(output_path)
                    emit_image_ready(run_id, output_path.name)
                elif enhance and enhance_after:
                    images_to_enhance.append((output_path, current_seed))

                if current_seed is not None:
                    current_seed += 1

        # Batch enhancement
        if enhance and enhance_after and images_to_enhance:
            clear_model_cache()
            emit_progress("enhancing_images", 0, len(images_to_enhance), "Starting batch enhancement...")

            for idx, (image_path, image_seed) in enumerate(images_to_enhance, 1):
                emit_progress(
                    "enhancing_images",
                    idx,
                    len(images_to_enhance),
                    f"Enhancing {image_path.name}..."
                )
                try:
                    with Heartbeat(f"Enhancing {image_path.name}..."):
                        enhance_image(
                            image_path=image_path,
                            output_path=image_path,
                            softness=enhance_softness,
                            seed=image_seed,
                            quantize=quantize,
                            tiled_vae=tiled_vae,
                        )
                except Exception as e:
                    emit_progress("error_detail", 0, 0, traceback.format_exc())
                    emit_result(False, error=f"Enhancement failed ({image_path.name}): {e}")
                    return

                sync_file(image_path)
                emit_image_ready(run_id, image_path.name)

    emit_result(True, data={
        "run_id": run_id,
        "prompt_count": len(outputs),
        "output_dir": str(output_dir),
    })


def run_regenerate_prompts(params: dict):
    """Regenerate prompts from edited grammar."""
    run_id = params["run_id"]
    grammar = params["grammar"]
    count = params.get("count")

    output_dir = GENERATED_DIR / "prompts" / run_id

    if not output_dir.exists():
        emit_result(False, error=f"Run directory not found: {run_id}")
        return

    # Load existing metadata
    meta_files = list(output_dir.glob("*_metadata.json"))
    if not meta_files:
        emit_result(False, error="No metadata file found")
        return

    metadata = json.loads(meta_files[0].read_text())
    prefix = metadata.get("prefix", "image")

    # Set up per-task log file
    set_log_file(output_dir / f"{prefix}_worker.log")
    log_to_file(f"Regenerating prompts: count={count}")

    if count is None:
        count = metadata.get("count", 50)

    emit_progress("expanding_prompts", 0, count, f"Regenerating {count} prompts...")

    try:
        outputs = run_tracery(grammar, count=count)
    except TraceryError as e:
        emit_result(False, error=f"Tracery expansion failed: {e}")
        return

    # Delete old prompt files
    for old_file in output_dir.glob(f"{prefix}_*.txt"):
        if old_file.stem.count('_') == 1:  # Only prompts, not metadata
            old_file.unlink()

    # Save new prompts
    for i, text in enumerate(outputs):
        output_file = output_dir / f"{prefix}_{i}.txt"
        output_file.write_text(text)

    # Update grammar file
    grammar_file = output_dir / f"{prefix}_grammar.json"
    grammar_file.write_text(grammar)

    # Update metadata
    metadata["count"] = len(outputs)
    metadata["regenerated_at"] = datetime.now().isoformat()
    meta_files[0].write_text(json.dumps(metadata, indent=2))

    emit_progress("expanding_prompts", count, count, f"Generated {len(outputs)} prompts")
    emit_result(True, data={
        "run_id": run_id,
        "prompt_count": len(outputs),
    })


def run_generate_image(params: dict):
    """Generate a single image."""
    run_id = params["run_id"]
    prompt_idx = params["prompt_idx"]
    image_idx = params.get("image_idx", 0)

    output_dir = GENERATED_DIR / "prompts" / run_id

    if not output_dir.exists():
        emit_result(False, error=f"Run directory not found: {run_id}")
        return

    # Load metadata
    meta_files = list(output_dir.glob("*_metadata.json"))
    if not meta_files:
        emit_result(False, error="No metadata file found")
        return

    metadata = json.loads(meta_files[0].read_text())
    prefix = metadata.get("prefix", "image")
    image_settings = metadata.get("image_generation", {})

    # Set up per-task log file
    set_log_file(output_dir / f"{prefix}_worker.log")
    log_to_file(f"Generating single image: prompt_idx={prompt_idx}, image_idx={image_idx}")

    # Load prompt
    prompt_file = output_dir / f"{prefix}_{prompt_idx}.txt"
    if not prompt_file.exists():
        emit_result(False, error=f"Prompt file not found: {prompt_file.name}")
        return

    prompt_text = prompt_file.read_text()
    output_path = output_dir / f"{prefix}_{prompt_idx}_{image_idx}.png"

    emit_progress("generating_image", 0, 1, f"Generating {output_path.name}...")

    model = image_settings.get("model", "z-image-turbo")
    try:
        with Heartbeat(f"Generating {output_path.name}..."):
            generate_image(
                prompt=prompt_text,
                output_path=output_path,
                model=model,
                seed=image_settings.get("seed"),
                steps=image_settings.get("steps"),
                width=image_settings.get("width", 864),
                height=image_settings.get("height", 1152),
                quantize=image_settings.get("quantize", 8),
                tiled_vae=True,
            )
    except Exception as e:
        context = f"model={model}, run_id={run_id}, prompt_idx={prompt_idx}, image_idx={image_idx}"
        emit_progress("error_detail", 0, 0, traceback.format_exc())
        emit_result(False, error=f"Image generation failed ({context}): {e}")
        return

    sync_file(output_path)
    emit_image_ready(run_id, output_path.name)
    emit_progress("generating_image", 1, 1, "Image generated")
    emit_result(True, data={
        "run_id": run_id,
        "image_path": output_path.name,
    })


def run_enhance_image(params: dict):
    """Enhance a single image."""
    run_id = params["run_id"]
    prompt_idx = params["prompt_idx"]
    image_idx = params.get("image_idx", 0)
    softness = params.get("softness", 0.5)

    output_dir = GENERATED_DIR / "prompts" / run_id

    if not output_dir.exists():
        emit_result(False, error=f"Run directory not found: {run_id}")
        return

    # Load metadata
    meta_files = list(output_dir.glob("*_metadata.json"))
    if not meta_files:
        emit_result(False, error="No metadata file found")
        return

    metadata = json.loads(meta_files[0].read_text())
    prefix = metadata.get("prefix", "image")
    image_settings = metadata.get("image_generation", {})

    # Set up per-task log file
    set_log_file(output_dir / f"{prefix}_worker.log")
    log_to_file(f"Enhancing single image: prompt_idx={prompt_idx}, image_idx={image_idx}")

    image_path = output_dir / f"{prefix}_{prompt_idx}_{image_idx}.png"
    if not image_path.exists():
        emit_result(False, error=f"Image not found: {image_path.name}")
        return

    emit_progress("enhancing_image", 0, 1, f"Enhancing {image_path.name}...")

    try:
        with Heartbeat(f"Enhancing {image_path.name}..."):
            enhance_image(
                image_path=image_path,
                output_path=image_path,
                softness=softness,
                quantize=image_settings.get("quantize", 8),
                tiled_vae=True,
            )
    except Exception as e:
        context = f"run_id={run_id}, prompt_idx={prompt_idx}, image_idx={image_idx}"
        emit_progress("error_detail", 0, 0, traceback.format_exc())
        emit_result(False, error=f"Enhancement failed ({context}): {e}")
        return

    sync_file(image_path)
    emit_image_ready(run_id, image_path.name)
    emit_progress("enhancing_image", 1, 1, "Enhancement complete")
    emit_result(True, data={
        "run_id": run_id,
        "image_path": image_path.name,
    })


def run_generate_all_images(params: dict):
    """Generate all images for a gallery."""
    run_id = params["run_id"]
    images_per_prompt = params.get("images_per_prompt", 1)
    resume = params.get("resume", True)

    output_dir = GENERATED_DIR / "prompts" / run_id

    if not output_dir.exists():
        emit_result(False, error=f"Run directory not found: {run_id}")
        return

    # Load metadata
    meta_files = list(output_dir.glob("*_metadata.json"))
    if not meta_files:
        emit_result(False, error="No metadata file found")
        return

    metadata = json.loads(meta_files[0].read_text())
    prefix = metadata.get("prefix", "image")
    image_settings = metadata.get("image_generation", {})

    # Set up per-task log file
    set_log_file(output_dir / f"{prefix}_worker.log")
    log_to_file(f"Generating all images: images_per_prompt={images_per_prompt}, resume={resume}")

    # Load prompts
    prompt_files = sorted(output_dir.glob(f"{prefix}_*.txt"))
    prompt_files = [f for f in prompt_files if f.stem.count('_') == 1]

    if not prompt_files:
        emit_result(False, error="No prompt files found")
        return

    prompts = [f.read_text() for f in prompt_files]
    total_images = len(prompts) * images_per_prompt

    emit_progress("generating_images", 0, total_images, "Starting image generation...")

    current_seed = image_settings.get("seed")
    generated_count = 0
    skipped_count = 0

    for prompt_idx, prompt_text in enumerate(prompts):
        for image_idx in range(images_per_prompt):
            output_path = output_dir / f"{prefix}_{prompt_idx}_{image_idx}.png"

            if resume and output_path.exists():
                skipped_count += 1
                generated_count += 1
                if current_seed is not None:
                    current_seed += 1
                continue

            generated_count += 1
            emit_progress(
                "generating_images",
                generated_count,
                total_images,
                f"Generating {output_path.name}..."
            )

            model = image_settings.get("model", "z-image-turbo")
            try:
                with Heartbeat(f"Generating {output_path.name}..."):
                    generate_image(
                        prompt=prompt_text,
                        output_path=output_path,
                        model=model,
                        seed=current_seed,
                        steps=image_settings.get("steps"),
                        width=image_settings.get("width", 864),
                        height=image_settings.get("height", 1152),
                        quantize=image_settings.get("quantize", 8),
                        tiled_vae=True,
                    )
            except Exception as e:
                context = f"model={model}, run_id={run_id}, prompt_idx={prompt_idx}, image_idx={image_idx}"
                emit_progress("error_detail", 0, 0, traceback.format_exc())
                emit_result(False, error=f"Image generation failed ({context}): {e}")
                return

            sync_file(output_path)
            emit_image_ready(run_id, output_path.name)

            if current_seed is not None:
                current_seed += 1

    actual_generated = generated_count - skipped_count
    emit_result(True, data={
        "run_id": run_id,
        "generated": actual_generated,
        "skipped": skipped_count,
    })


def run_enhance_all_images(params: dict):
    """Enhance all images for a gallery."""
    run_id = params["run_id"]
    softness = params.get("softness", 0.5)

    output_dir = GENERATED_DIR / "prompts" / run_id

    if not output_dir.exists():
        emit_result(False, error=f"Run directory not found: {run_id}")
        return

    # Load metadata
    meta_files = list(output_dir.glob("*_metadata.json"))
    if not meta_files:
        emit_result(False, error="No metadata file found")
        return

    metadata = json.loads(meta_files[0].read_text())
    prefix = metadata.get("prefix", "image")
    image_settings = metadata.get("image_generation", {})

    # Set up per-task log file
    set_log_file(output_dir / f"{prefix}_worker.log")
    log_to_file(f"Enhancing all images: softness={softness}")

    # Find all images
    images = sorted(output_dir.glob(f"{prefix}_*_*.png"))
    if not images:
        emit_result(False, error="No images found")
        return

    emit_progress("enhancing_images", 0, len(images), "Starting enhancement...")

    for idx, image_path in enumerate(images, 1):
        emit_progress(
            "enhancing_images",
            idx,
            len(images),
            f"Enhancing {image_path.name}..."
        )

        try:
            with Heartbeat(f"Enhancing {image_path.name}..."):
                enhance_image(
                    image_path=image_path,
                    output_path=image_path,
                    softness=softness,
                    quantize=image_settings.get("quantize", 8),
                    tiled_vae=True,
                )
        except Exception as e:
            emit_progress("error_detail", 0, 0, traceback.format_exc())
            emit_result(False, error=f"Enhancement failed ({image_path.name}): {e}")
            return

        sync_file(image_path)
        emit_image_ready(run_id, image_path.name)

    emit_result(True, data={
        "run_id": run_id,
        "enhanced": len(images),
    })


TASK_HANDLERS = {
    "generate_pipeline": run_generate_pipeline,
    "regenerate_prompts": run_regenerate_prompts,
    "generate_image": run_generate_image,
    "enhance_image": run_enhance_image,
    "generate_all_images": run_generate_all_images,
    "enhance_all_images": run_enhance_all_images,
}


def main():
    if len(sys.argv) != 2:
        print("Usage: python worker_subprocess.py <task.json>", file=sys.stderr)
        sys.exit(1)

    task_file = Path(sys.argv[1])
    if not task_file.exists():
        print(f"Task file not found: {task_file}", file=sys.stderr)
        sys.exit(1)

    try:
        task = json.loads(task_file.read_text())
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in task file: {e}", file=sys.stderr)
        sys.exit(1)

    task_type = task.get("type")
    params = task.get("params", {})

    handler = TASK_HANDLERS.get(task_type)
    if handler is None:
        emit_result(False, error=f"Unknown task type: {task_type}")
        sys.exit(1)

    try:
        handler(params)
    except Exception as e:
        log_to_file(f"Unhandled exception: {traceback.format_exc()}")
        emit_result(False, error=str(e))
        sys.exit(1)
    finally:
        close_log_file()


if __name__ == "__main__":
    main()
