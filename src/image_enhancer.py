"""SeedVR2 image enhancement using mflux (MLX-based for Apple Silicon)."""

import random
from pathlib import Path


# Cache for loaded enhancer model (expensive to load)
_enhancer_cache: dict = {}


def clear_enhancer_cache():
    """Clear the enhancer cache and free memory."""
    global _enhancer_cache
    _enhancer_cache.clear()
    import gc
    gc.collect()


def _get_enhancer(quantize: int, tiled_vae: bool = False):
    """Get or create a cached SeedVR2 enhancer instance."""
    cache_key = (quantize, tiled_vae)
    if cache_key in _enhancer_cache:
        return _enhancer_cache[cache_key]

    try:
        from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
    except ImportError as e:
        raise ImportError(
            "mflux is required for image enhancement. "
            "Install with: pip install mflux\n"
            "Note: mflux requires macOS with Apple Silicon (M1/M2/M3/M4)."
        ) from e

    instance = SeedVR2(quantize=quantize)

    # SeedVR2 has tiling enabled by default; disable if requested
    if not tiled_vae:
        instance.tiling_config = None

    _enhancer_cache[cache_key] = instance
    return instance


def enhance_image(
    image_path: Path,
    output_path: Path,
    softness: float = 0.5,
    seed: int | None = None,
    quantize: int = 8,
    tiled_vae: bool = False,
) -> Path:
    """
    Enhance a single image using SeedVR2 2x upscaling.

    Args:
        image_path: Path to the source image
        output_path: Path where the enhanced image will be saved
        softness: Enhancement softness (0.0-1.0, default 0.5)
        seed: Random seed for reproducibility (None for random)
        quantize: Quantization level (3, 4, 5, 6, or 8)
        tiled_vae: Enable tiled VAE decoding to reduce memory (default: True)

    Returns:
        Path to the enhanced image file

    Raises:
        ImportError: If mflux is not installed
        FileNotFoundError: If image_path does not exist
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Import ScaleFactor for 2x upscaling
    try:
        from mflux.utils.scale_factor import ScaleFactor
    except ImportError as e:
        raise ImportError(
            "mflux is required for image enhancement. "
            "Install with: pip install mflux"
        ) from e

    # Generate random seed if not specified
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get or create the enhancer
    enhancer = _get_enhancer(quantize, tiled_vae)

    # Enhance the image with 2x upscaling
    result = enhancer.generate_image(
        seed=seed,
        image_path=str(image_path),
        resolution=ScaleFactor(2),
        softness=softness,
    )

    # Save the enhanced image (overwrite if replacing original)
    result.save(path=str(output_path), overwrite=True)

    return output_path


def enhance_images(
    image_paths: list[Path],
    output_dir: Path | None = None,
    softness: float = 0.5,
    seed: int | None = None,
    quantize: int = 8,
    in_place: bool = True,
    on_progress: callable = None,
) -> list[Path]:
    """
    Enhance multiple images using SeedVR2.

    Args:
        image_paths: List of image paths to enhance
        output_dir: Output directory (None = same directory as source)
        softness: Enhancement softness (0.0-1.0, default 0.5)
        seed: Base random seed (incremented for each image)
        quantize: Quantization level
        in_place: If True, replace original files (default). If False, use output_dir.
        on_progress: Callback function(current, total)

    Returns:
        List of paths to enhanced images
    """
    enhanced_paths = []
    current_seed = seed
    total = len(image_paths)

    for idx, image_path in enumerate(image_paths):
        # Determine output path
        if in_place:
            output_path = image_path
        elif output_dir is not None:
            output_path = output_dir / image_path.name
        else:
            output_path = image_path

        if on_progress:
            on_progress(idx + 1, total)

        enhance_image(
            image_path=image_path,
            output_path=output_path,
            softness=softness,
            seed=current_seed,
            quantize=quantize,
        )

        enhanced_paths.append(output_path)

        # Increment seed for next image (if seed was specified)
        if current_seed is not None:
            current_seed += 1

    return enhanced_paths


def collect_images(path_spec: str) -> list[Path]:
    """
    Collect image paths from a file, directory, or glob pattern.

    Args:
        path_spec: Path to file, directory, or glob pattern

    Returns:
        List of image paths (sorted)

    Raises:
        ValueError: If no images found
    """
    from glob import glob

    path = Path(path_spec)
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}

    if path.is_file():
        # Single file
        if path.suffix.lower() in image_extensions:
            return [path]
        raise ValueError(f"Not an image file: {path}")

    elif path.is_dir():
        # Directory - find all images
        images = []
        for ext in image_extensions:
            images.extend(path.glob(f"*{ext}"))
            images.extend(path.glob(f"*{ext.upper()}"))
        images = sorted(set(images))
        if not images:
            raise ValueError(f"No images found in directory: {path}")
        return images

    else:
        # Treat as glob pattern
        matches = glob(path_spec)
        images = [Path(m) for m in matches if Path(m).suffix.lower() in image_extensions]
        images = sorted(set(images))
        if not images:
            raise ValueError(f"No images found matching pattern: {path_spec}")
        return images
