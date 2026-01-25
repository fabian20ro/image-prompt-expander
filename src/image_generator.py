"""Image generation wrapper for mflux (MLX-based FLUX for Apple Silicon)."""

import random
from pathlib import Path


# Default steps per model (based on mflux recommendations)
MODEL_DEFAULTS = {
    "z-image-turbo": {"steps": 9},
    "flux2-klein-4b": {"steps": 4},
    "flux2-klein-9b": {"steps": 4},
}

# Pre-quantized model paths on HuggingFace
MODEL_PATHS = {
    "z-image-turbo": "filipstrand/Z-Image-Turbo-mflux-4bit",
    "flux2-klein-4b": None,  # No pre-quantized version available
    "flux2-klein-9b": None,  # No pre-quantized version available
}

SUPPORTED_MODELS = list(MODEL_DEFAULTS.keys())

# Cache for loaded models (expensive to load)
_model_cache: dict = {}


def clear_model_cache():
    """Clear the model cache and free memory."""
    global _model_cache
    _model_cache.clear()
    import gc
    gc.collect()


def _get_model(model: str, quantize: int, tiled_vae: bool = True):
    """Get or create a cached model instance."""
    cache_key = (model, quantize, tiled_vae)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        from mflux.models.common.config.model_config import ModelConfig
    except ImportError as e:
        raise ImportError(
            "mflux is required for image generation. "
            "Install with: pip install mflux\n"
            "Note: mflux requires macOS with Apple Silicon (M1/M2/M3/M4)."
        ) from e

    # Use pre-quantized model path if available, otherwise use default
    model_path = MODEL_PATHS.get(model)

    if model == "z-image-turbo":
        from mflux.models.z_image import ZImageTurbo
        instance = ZImageTurbo(
            quantize=quantize if model_path is None else None,
            model_path=model_path,
            model_config=ModelConfig.z_image_turbo(),
        )
    elif model == "flux2-klein-4b":
        from mflux.models.flux2 import Flux2Klein
        instance = Flux2Klein(
            quantize=None,  # Use full model without quantization
            model_path=model_path,
            model_config=ModelConfig.flux2_klein_4b(),
        )
    elif model == "flux2-klein-9b":
        from mflux.models.flux2 import Flux2Klein
        instance = Flux2Klein(
            quantize=None,  # Use full model without quantization
            model_path=model_path,
            model_config=ModelConfig.flux2_klein_9b(),
        )
    else:
        raise ValueError(f"Unsupported model: {model}. Choose from: {SUPPORTED_MODELS}")

    # Enable tiled VAE decoding for reduced memory usage
    if tiled_vae:
        from mflux.models.common.vae.tiling_config import TilingConfig
        instance.tiling_config = TilingConfig()

    _model_cache[cache_key] = instance
    return instance


def generate_image(
    prompt: str,
    output_path: Path,
    model: str = "z-image-turbo",
    seed: int | None = None,
    steps: int | None = None,
    width: int = 864,
    height: int = 1152,
    quantize: int = 8,
    tiled_vae: bool = True,
) -> Path:
    """
    Generate a single image from a prompt using mflux.

    Args:
        prompt: Text description for the image
        output_path: Path where the PNG will be saved
        model: Model to use (z-image-turbo, flux2-klein-4b, flux2-klein-9b)
        seed: Random seed for reproducibility (None for random)
        steps: Number of inference steps (None uses model-specific default)
        width: Image width in pixels (default 1024)
        height: Image height in pixels (default 1024)
        quantize: Quantization level (3, 4, 5, 6, or 8)
        tiled_vae: Enable tiled VAE decoding to reduce memory (default: True)

    Returns:
        Path to the generated image file

    Raises:
        ImportError: If mflux is not installed
        ValueError: If model is not supported
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}. Choose from: {SUPPORTED_MODELS}")

    # Use model-specific default steps if not specified
    if steps is None:
        steps = MODEL_DEFAULTS[model]["steps"]

    # Generate random seed if not specified
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get or create the model
    flux = _get_model(model, quantize, tiled_vae)

    # Generate the image
    if model == "z-image-turbo":
        # ZImageTurbo returns PIL.Image.Image directly
        image = flux.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
        )
        # Save PIL image
        image.save(str(output_path))
    else:
        # Flux2Klein returns GeneratedImage
        result = flux.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
        )
        # Save using GeneratedImage.save()
        result.save(path=str(output_path))

    return output_path


def generate_images_for_prompts(
    prompts: list[str],
    output_dir: Path,
    prefix: str = "image",
    images_per_prompt: int = 1,
    model: str = "z-image-turbo",
    seed: int | None = None,
    steps: int | None = None,
    width: int = 864,
    height: int = 1152,
    quantize: int = 8,
    max_prompts: int | None = None,
    on_progress: callable = None,
) -> list[Path]:
    """
    Generate images for multiple prompts.

    Args:
        prompts: List of text prompts
        output_dir: Directory for output files
        prefix: Prefix for output filenames
        images_per_prompt: Number of images to generate per prompt
        model: Model to use
        seed: Base random seed (incremented for each image)
        steps: Number of inference steps
        width: Image width
        height: Image height
        quantize: Quantization level
        max_prompts: Maximum number of prompts to process (None for all)
        on_progress: Callback function(prompt_idx, image_idx, total_images)

    Returns:
        List of paths to generated images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []

    # Limit prompts if max_prompts is set
    prompts_to_process = prompts[:max_prompts] if max_prompts else prompts
    total_images = len(prompts_to_process) * images_per_prompt

    current_seed = seed
    image_count = 0

    for prompt_idx, prompt in enumerate(prompts_to_process):
        for image_idx in range(images_per_prompt):
            output_path = output_dir / f"{prefix}_{prompt_idx}_{image_idx}.png"

            if on_progress:
                on_progress(prompt_idx, image_idx, total_images)

            generate_image(
                prompt=prompt,
                output_path=output_path,
                model=model,
                seed=current_seed,
                steps=steps,
                width=width,
                height=height,
                quantize=quantize,
            )

            generated_paths.append(output_path)
            image_count += 1

            # Increment seed for next image (if seed was specified)
            if current_seed is not None:
                current_seed += 1

    return generated_paths
