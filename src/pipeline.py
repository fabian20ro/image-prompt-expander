"""Pipeline executor for image prompt generation.

Provides a unified interface for all pipeline operations that can be
used by both the CLI and web UI. Uses callback-based progress reporting.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

logger = logging.getLogger(__name__)

from config import paths
from grammar_generator import generate_grammar, hash_prompt
from tracery_runner import run_tracery, TraceryError
from image_generator import generate_image, clear_model_cache, MODEL_DEFAULTS
from image_enhancer import enhance_image
from gallery import create_gallery, update_gallery
from gallery_index import generate_master_index
from utils import backup_run, run_has_images


class ProgressCallback(Protocol):
    """Protocol for progress callbacks."""

    def __call__(
        self,
        stage: str,
        current: int = 0,
        total: int = 0,
        message: str = "",
    ) -> None:
        """Report progress.

        Args:
            stage: Current stage (e.g., "generating_grammar", "generating_images")
            current: Current progress count
            total: Total items to process
            message: Human-readable progress message
        """
        ...


class ImageReadyCallback(Protocol):
    """Protocol for image ready notifications."""

    def __call__(self, run_id: str, filename: str) -> None:
        """Notify that an image is ready.

        Args:
            run_id: The run directory name
            filename: The image filename
        """
        ...


@dataclass
class PipelineResult:
    """Result of a pipeline operation."""

    success: bool
    run_id: str | None = None
    output_dir: Path | None = None
    prompt_count: int = 0
    image_count: int = 0
    skipped_count: int = 0
    error: str | None = None
    data: dict = field(default_factory=dict)


@dataclass
class ImageGenerationConfig:
    """Configuration for image generation settings.

    Groups all image-related parameters to reduce function signature complexity.
    """

    enabled: bool = False
    images_per_prompt: int = 1
    width: int = 864
    height: int = 1152
    steps: int | None = None  # Uses model default if None
    quantize: int = 8
    seed: int | None = None
    max_prompts: int | None = None  # Limit prompts to render
    tiled_vae: bool = True
    resume: bool = False  # Skip existing images

    def to_dict(self) -> dict:
        """Convert to dictionary for metadata storage."""
        result = {
            "enabled": self.enabled,
            "images_per_prompt": self.images_per_prompt,
            "width": self.width,
            "height": self.height,
            "quantize": self.quantize,
            "tiled_vae": self.tiled_vae,
        }
        if self.steps is not None:
            result["steps"] = self.steps
        if self.seed is not None:
            result["seed"] = self.seed
        if self.max_prompts is not None:
            result["max_prompts"] = self.max_prompts
        return result


@dataclass
class EnhancementConfig:
    """Configuration for image enhancement settings."""

    enabled: bool = False
    softness: float = 0.5
    batch_after: bool = False  # Enhance all images after generation completes

    def to_dict(self) -> dict:
        """Convert to dictionary for metadata storage."""
        return {
            "enabled": self.enabled,
            "softness": self.softness,
            "batch_after": self.batch_after,
        }


@dataclass
class PipelineConfig:
    """Complete configuration for pipeline execution.

    This class groups all pipeline parameters into logical categories,
    reducing function signature complexity from 20+ parameters to a single config.

    Example:
        config = PipelineConfig(
            prompt="a dragon flying",
            count=50,
            image=ImageGenerationConfig(enabled=True, width=1024, height=1024),
            enhancement=EnhancementConfig(enabled=True, softness=0.3),
        )
        result = executor.run_with_config(config)
    """

    # Required
    prompt: str = ""

    # Grammar/Prompt settings
    count: int = 50
    prefix: str = "image"
    model: str = "z-image-turbo"  # Used for both grammar gen and image gen
    temperature: float = 0.7
    no_cache: bool = False

    # Image generation settings
    image: ImageGenerationConfig = field(default_factory=ImageGenerationConfig)

    # Enhancement settings
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)

    # Output settings
    output_dir: Path | None = None

    @classmethod
    def from_kwargs(cls, **kwargs) -> "PipelineConfig":
        """Create config from flat kwargs (for backwards compatibility).

        Maps old-style parameters to the new config structure.
        """
        # Extract image config params
        image_config = ImageGenerationConfig(
            enabled=kwargs.pop("generate_images", False),
            images_per_prompt=kwargs.pop("images_per_prompt", 1),
            width=kwargs.pop("width", 864),
            height=kwargs.pop("height", 1152),
            steps=kwargs.pop("steps", None),
            quantize=kwargs.pop("quantize", 8),
            seed=kwargs.pop("seed", None),
            max_prompts=kwargs.pop("max_prompts", None),
            tiled_vae=kwargs.pop("tiled_vae", True),
            resume=kwargs.pop("resume", False),
        )

        # Extract enhancement config params
        enhancement_config = EnhancementConfig(
            enabled=kwargs.pop("enhance", False),
            softness=kwargs.pop("enhance_softness", 0.5),
            batch_after=kwargs.pop("enhance_after", False),
        )

        return cls(
            prompt=kwargs.pop("prompt", ""),
            count=kwargs.pop("count", 50),
            prefix=kwargs.pop("prefix", "image"),
            model=kwargs.pop("model", "z-image-turbo"),
            temperature=kwargs.pop("temperature", 0.7),
            no_cache=kwargs.pop("no_cache", False),
            image=image_config,
            enhancement=enhancement_config,
            output_dir=kwargs.pop("output_dir", None),
        )


def _null_progress(stage: str, current: int = 0, total: int = 0, message: str = "") -> None:
    """No-op progress callback."""
    pass


def _null_image_ready(run_id: str, filename: str) -> None:
    """No-op image ready callback."""
    pass


class PipelineExecutor:
    """Executor for image prompt generation pipeline operations."""

    def __init__(
        self,
        on_progress: ProgressCallback | None = None,
        on_image_ready: ImageReadyCallback | None = None,
    ):
        """Initialize the executor.

        Args:
            on_progress: Callback for progress updates
            on_image_ready: Callback when an image is ready
        """
        self.on_progress = on_progress or _null_progress
        self.on_image_ready = on_image_ready or _null_image_ready

    def run_full_pipeline(
        self,
        prompt: str,
        count: int = 50,
        prefix: str = "image",
        model: str = "z-image-turbo",
        temperature: float = 0.7,
        no_cache: bool = False,
        generate_images: bool = False,
        images_per_prompt: int = 1,
        width: int = 864,
        height: int = 1152,
        steps: int | None = None,
        quantize: int = 8,
        seed: int | None = None,
        max_prompts: int | None = None,
        tiled_vae: bool = True,
        enhance: bool = False,
        enhance_softness: float = 0.5,
        enhance_after: bool = False,
        resume: bool = False,
        output_dir: Path | None = None,
    ) -> PipelineResult:
        """Run the full generation pipeline: LLM -> Tracery -> Images.

        Args:
            prompt: User's image description
            count: Number of prompt variations to generate
            prefix: Prefix for output files
            model: Image model to use
            temperature: LLM temperature for grammar generation
            no_cache: Skip grammar cache
            generate_images: Whether to generate images
            images_per_prompt: Number of images per prompt
            width: Image width
            height: Image height
            steps: Inference steps (model-specific default if None)
            quantize: Quantization level
            seed: Random seed
            max_prompts: Limit prompts to render
            tiled_vae: Use tiled VAE decoding
            enhance: Enable SeedVR2 enhancement
            enhance_softness: Enhancement softness
            enhance_after: Batch enhancement after all images
            resume: Skip existing images
            output_dir: Custom output directory

        Returns:
            PipelineResult with operation results
        """
        # Stage 1: Generate grammar
        cache_msg = " (ignoring cache)" if no_cache else ""
        self.on_progress(
            "generating_grammar", 0, 1,
            f"Generating grammar{cache_msg} for: {prompt[:50]}..."
        )

        try:
            grammar, was_cached, raw_response = generate_grammar(
                user_prompt=prompt,
                use_cache=not no_cache,
                temperature=temperature,
                model=model,
            )
        except Exception as e:
            return PipelineResult(success=False, error=f"Grammar generation failed: {e}")

        self.on_progress("generating_grammar", 1, 1, "Grammar generated")

        # Stage 2: Run Tracery
        self.on_progress("expanding_prompts", 0, count, f"Expanding {count} prompts...")

        try:
            outputs = run_tracery(grammar, count=count)
        except TraceryError as e:
            return PipelineResult(success=False, error=f"Tracery expansion failed: {e}")

        self.on_progress("expanding_prompts", count, count, f"Generated {len(outputs)} prompts")

        # Create output directory
        if output_dir is None:
            prompt_hash = hash_prompt(prompt)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = paths.prompts_dir / f"{timestamp}_{prompt_hash}"

        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = output_dir.name

        # Save prompts
        for i, text in enumerate(outputs):
            output_file = output_dir / f"{prefix}_{i}.txt"
            output_file.write_text(text)

        # Build metadata
        effective_steps = steps if steps is not None else MODEL_DEFAULTS[model]["steps"]
        metadata = {
            "user_prompt": prompt,
            "count": len(outputs),
            "created_at": datetime.now().isoformat(),
            "grammar_cached": was_cached,
            "prefix": prefix,
            "model": model,
        }

        if generate_images:
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

        # Create gallery
        prompts_to_render = outputs[:max_prompts] if max_prompts else outputs
        gallery_path = create_gallery(
            output_dir, prefix, prompts_to_render,
            images_per_prompt if generate_images else 0,
            grammar=grammar, raw_response_file=raw_response_file
        )

        # Update master index
        generate_master_index(paths.generated_dir)

        # Stage 3: Generate images (if requested)
        generated_count = 0
        skipped_count = 0

        if generate_images:
            result = self._generate_images(
                output_dir=output_dir,
                run_id=run_id,
                prefix=prefix,
                prompts=prompts_to_render,
                images_per_prompt=images_per_prompt,
                model=model,
                steps=steps,
                width=width,
                height=height,
                quantize=quantize,
                seed=seed,
                tiled_vae=tiled_vae,
                enhance=enhance,
                enhance_softness=enhance_softness,
                enhance_after=enhance_after,
                resume=resume,
                gallery_path=gallery_path,
            )
            if not result.success:
                return result
            generated_count = result.image_count
            skipped_count = result.skipped_count

        return PipelineResult(
            success=True,
            run_id=run_id,
            output_dir=output_dir,
            prompt_count=len(outputs),
            image_count=generated_count,
            skipped_count=skipped_count,
        )

    def run_from_grammar(
        self,
        grammar_path: Path,
        count: int = 50,
        prefix: str = "image",
        model: str = "z-image-turbo",
        generate_images: bool = False,
        images_per_prompt: int = 1,
        width: int = 864,
        height: int = 1152,
        steps: int | None = None,
        quantize: int = 8,
        seed: int | None = None,
        max_prompts: int | None = None,
        tiled_vae: bool = True,
        enhance: bool = False,
        enhance_softness: float = 0.5,
        enhance_after: bool = False,
        resume: bool = False,
        output_dir: Path | None = None,
    ) -> PipelineResult:
        """Run pipeline from an existing grammar file.

        Args:
            grammar_path: Path to existing grammar file
            count: Number of prompt variations
            prefix: Prefix for output files
            model: Image model for generation
            generate_images: Whether to generate images
            images_per_prompt: Images per prompt
            width: Image width
            height: Image height
            steps: Inference steps
            quantize: Quantization level
            seed: Random seed
            max_prompts: Limit prompts to render
            tiled_vae: Use tiled VAE
            enhance: Enable enhancement
            enhance_softness: Enhancement softness
            enhance_after: Batch enhancement
            resume: Skip existing images
            output_dir: Custom output directory

        Returns:
            PipelineResult with operation results
        """
        grammar = grammar_path.read_text()

        # Try to find metadata for original prompt
        meta_path = grammar_path.with_suffix('.metaprompt.json')
        user_prompt = "unknown"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            user_prompt = meta.get("user_prompt", "unknown")

        # Try to find raw response
        grammar_hash = grammar_path.stem.replace('.tracery', '')
        raw_path = grammar_path.parent / f"{grammar_hash}.raw.txt"
        raw_response = raw_path.read_text() if raw_path.exists() else None

        # Run Tracery
        self.on_progress("expanding_prompts", 0, count, f"Expanding {count} prompts...")

        try:
            outputs = run_tracery(grammar, count=count)
        except TraceryError as e:
            return PipelineResult(success=False, error=f"Tracery expansion failed: {e}")

        self.on_progress("expanding_prompts", count, count, f"Generated {len(outputs)} prompts")

        # Create output directory
        if output_dir is None:
            grammar_stem = grammar_path.stem.replace('.tracery', '')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = paths.prompts_dir / f"{timestamp}_{grammar_stem}"

        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = output_dir.name

        # Save prompts
        for i, text in enumerate(outputs):
            output_file = output_dir / f"{prefix}_{i}.txt"
            output_file.write_text(text)

        # Build metadata
        effective_steps = steps if steps is not None else MODEL_DEFAULTS[model]["steps"]
        metadata = {
            "source": "from_grammar",
            "grammar_path": str(grammar_path),
            "user_prompt": user_prompt,
            "count": len(outputs),
            "created_at": datetime.now().isoformat(),
            "grammar_cached": True,
            "prefix": prefix,
            "model": model,
        }

        if generate_images:
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

        # Save raw response if available
        raw_response_file = None
        if raw_response:
            raw_file = output_dir / f"{prefix}_raw_response.txt"
            raw_file.write_text(raw_response)
            raw_response_file = f"{prefix}_raw_response.txt"

        # Create gallery
        prompts_to_render = outputs[:max_prompts] if max_prompts else outputs
        gallery_path = create_gallery(
            output_dir, prefix, prompts_to_render,
            images_per_prompt if generate_images else 0,
            grammar=grammar, raw_response_file=raw_response_file
        )

        # Update master index
        generate_master_index(paths.generated_dir)

        # Generate images if requested
        generated_count = 0
        skipped_count = 0

        if generate_images:
            result = self._generate_images(
                output_dir=output_dir,
                run_id=run_id,
                prefix=prefix,
                prompts=prompts_to_render,
                images_per_prompt=images_per_prompt,
                model=model,
                steps=steps,
                width=width,
                height=height,
                quantize=quantize,
                seed=seed,
                tiled_vae=tiled_vae,
                enhance=enhance,
                enhance_softness=enhance_softness,
                enhance_after=enhance_after,
                resume=resume,
                gallery_path=gallery_path,
            )
            if not result.success:
                return result
            generated_count = result.image_count
            skipped_count = result.skipped_count

        return PipelineResult(
            success=True,
            run_id=run_id,
            output_dir=output_dir,
            prompt_count=len(outputs),
            image_count=generated_count,
            skipped_count=skipped_count,
        )

    def run_from_prompts(
        self,
        prompts_dir: Path,
        images_per_prompt: int = 1,
        model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        quantize: int | None = None,
        seed: int | None = None,
        max_prompts: int | None = None,
        tiled_vae: bool = True,
        enhance: bool = False,
        enhance_softness: float = 0.5,
        enhance_after: bool = False,
        resume: bool = False,
    ) -> PipelineResult:
        """Generate images from an existing prompts directory.

        Args:
            prompts_dir: Path to existing prompts directory
            images_per_prompt: Images per prompt
            model: Image model (uses metadata default if None)
            width: Image width (uses metadata default if None)
            height: Image height (uses metadata default if None)
            steps: Inference steps (uses metadata default if None)
            quantize: Quantization level (uses metadata default if None)
            seed: Random seed
            max_prompts: Limit prompts to render
            tiled_vae: Use tiled VAE
            enhance: Enable enhancement
            enhance_softness: Enhancement softness
            enhance_after: Batch enhancement
            resume: Skip existing images

        Returns:
            PipelineResult with operation results
        """
        # Load metadata
        meta_files = list(prompts_dir.glob("*_metadata.json"))
        if not meta_files:
            return PipelineResult(success=False, error=f"No metadata file found in {prompts_dir}")

        metadata = json.loads(meta_files[0].read_text())
        prefix = metadata.get("prefix", "image")
        existing_settings = metadata.get("image_generation", {})

        # Use existing settings as defaults
        model = model or existing_settings.get("model", "z-image-turbo")
        width = width or existing_settings.get("width", 864)
        height = height or existing_settings.get("height", 1152)
        steps = steps or existing_settings.get("steps")
        quantize = quantize or existing_settings.get("quantize", 8)

        # Load prompts
        prompt_files = sorted(prompts_dir.glob(f"{prefix}_*.txt"))
        prompt_files = [f for f in prompt_files if f.stem.count('_') == 1]

        if not prompt_files:
            return PipelineResult(success=False, error=f"No prompt files found in {prompts_dir}")

        outputs = [f.read_text() for f in prompt_files]
        prompts_to_render = outputs[:max_prompts] if max_prompts else outputs

        run_id = prompts_dir.name

        # Update metadata
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
            "resumed_from": str(prompts_dir),
            "original_settings": existing_settings if existing_settings else None,
            "enhance": enhance,
            "enhance_softness": enhance_softness if enhance else None,
        }
        meta_files[0].write_text(json.dumps(metadata, indent=2))

        # Load grammar for gallery
        grammar = None
        grammar_file = prompts_dir / f"{prefix}_grammar.json"
        if grammar_file.exists():
            grammar = grammar_file.read_text()

        raw_response_file = None
        raw_file = prompts_dir / f"{prefix}_raw_response.txt"
        if raw_file.exists():
            raw_response_file = f"{prefix}_raw_response.txt"

        # Create/update gallery
        gallery_path = create_gallery(
            prompts_dir, prefix, prompts_to_render, images_per_prompt,
            grammar=grammar, raw_response_file=raw_response_file
        )

        # Generate images
        result = self._generate_images(
            output_dir=prompts_dir,
            run_id=run_id,
            prefix=prefix,
            prompts=prompts_to_render,
            images_per_prompt=images_per_prompt,
            model=model,
            steps=steps,
            width=width,
            height=height,
            quantize=quantize,
            seed=seed,
            tiled_vae=tiled_vae,
            enhance=enhance,
            enhance_softness=enhance_softness,
            enhance_after=enhance_after,
            resume=resume,
            gallery_path=gallery_path,
        )

        return result

    def regenerate_prompts(
        self,
        run_id: str,
        grammar: str,
        count: int | None = None,
    ) -> PipelineResult:
        """Regenerate prompts from edited grammar.

        Args:
            run_id: Run directory name
            grammar: New grammar JSON
            count: Number of prompts (uses metadata default if None)

        Returns:
            PipelineResult with operation results
        """
        output_dir = paths.prompts_dir / run_id

        if not output_dir.exists():
            return PipelineResult(success=False, error=f"Run directory not found: {run_id}")

        # Load metadata
        meta_files = list(output_dir.glob("*_metadata.json"))
        if not meta_files:
            return PipelineResult(success=False, error="No metadata file found")

        metadata = json.loads(meta_files[0].read_text())
        prefix = metadata.get("prefix", "image")

        if count is None:
            count = metadata.get("count", 50)

        # Create backup if there are images
        if run_has_images(output_dir):
            self.on_progress("backup", 0, 1, "Creating backup before regenerating...")
            try:
                backup_run(output_dir, paths.saved_dir, reason="pre_regenerate")
            except Exception as e:
                logger.warning(f"Failed to create backup before regenerating: {e}")

        self.on_progress("expanding_prompts", 0, count, f"Regenerating {count} prompts...")

        try:
            outputs = run_tracery(grammar, count=count)
        except TraceryError as e:
            return PipelineResult(success=False, error=f"Tracery expansion failed: {e}")

        # Delete old prompt files
        for old_file in output_dir.glob(f"{prefix}_*.txt"):
            if old_file.stem.count('_') == 1:
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

        # Regenerate gallery
        self.on_progress("updating_gallery", 0, 1, "Updating gallery...")

        raw_response_file = None
        raw_file = output_dir / f"{prefix}_raw_response.txt"
        if raw_file.exists():
            raw_response_file = f"{prefix}_raw_response.txt"

        images_per_prompt = metadata.get("image_generation", {}).get("images_per_prompt", 1)
        if not metadata.get("image_generation", {}).get("enabled", False):
            images_per_prompt = 0

        create_gallery(
            output_dir, prefix, outputs, images_per_prompt,
            grammar=grammar, raw_response_file=raw_response_file,
            interactive=True, run_id=run_id
        )

        self.on_progress("updating_gallery", 1, 1, "Gallery updated")
        self.on_progress("expanding_prompts", count, count, f"Generated {len(outputs)} prompts")

        return PipelineResult(
            success=True,
            run_id=run_id,
            output_dir=output_dir,
            prompt_count=len(outputs),
            data={"task_type": "regenerate_prompts"},
        )

    def generate_single_image(
        self,
        run_id: str,
        prompt_idx: int,
        image_idx: int = 0,
    ) -> PipelineResult:
        """Generate a single image.

        Args:
            run_id: Run directory name
            prompt_idx: Index of the prompt
            image_idx: Index of the image for this prompt

        Returns:
            PipelineResult with operation results
        """
        output_dir = paths.prompts_dir / run_id

        if not output_dir.exists():
            return PipelineResult(success=False, error=f"Run directory not found: {run_id}")

        # Load metadata
        meta_files = list(output_dir.glob("*_metadata.json"))
        if not meta_files:
            return PipelineResult(success=False, error="No metadata file found")

        metadata = json.loads(meta_files[0].read_text())
        prefix = metadata.get("prefix", "image")
        image_settings = metadata.get("image_generation", {})

        # Load prompt
        prompt_file = output_dir / f"{prefix}_{prompt_idx}.txt"
        if not prompt_file.exists():
            return PipelineResult(success=False, error=f"Prompt file not found: {prompt_file.name}")

        prompt_text = prompt_file.read_text()
        output_path = output_dir / f"{prefix}_{prompt_idx}_{image_idx}.png"

        self.on_progress("generating_image", 0, 1, f"Generating {output_path.name}...")

        model = image_settings.get("model", "z-image-turbo")
        try:
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
            return PipelineResult(success=False, error=f"Image generation failed: {e}")

        self._sync_file(output_path)
        self.on_image_ready(run_id, output_path.name)
        self.on_progress("generating_image", 1, 1, "Image generated")

        return PipelineResult(
            success=True,
            run_id=run_id,
            image_count=1,
            data={"image_path": output_path.name},
        )

    def enhance_single_image(
        self,
        run_id: str,
        prompt_idx: int,
        image_idx: int = 0,
        softness: float = 0.5,
    ) -> PipelineResult:
        """Enhance a single image.

        Args:
            run_id: Run directory name
            prompt_idx: Index of the prompt
            image_idx: Index of the image
            softness: Enhancement softness

        Returns:
            PipelineResult with operation results
        """
        output_dir = paths.prompts_dir / run_id

        if not output_dir.exists():
            return PipelineResult(success=False, error=f"Run directory not found: {run_id}")

        # Load metadata
        meta_files = list(output_dir.glob("*_metadata.json"))
        if not meta_files:
            return PipelineResult(success=False, error="No metadata file found")

        metadata = json.loads(meta_files[0].read_text())
        prefix = metadata.get("prefix", "image")
        image_settings = metadata.get("image_generation", {})

        image_path = output_dir / f"{prefix}_{prompt_idx}_{image_idx}.png"
        if not image_path.exists():
            return PipelineResult(success=False, error=f"Image not found: {image_path.name}")

        self.on_progress("enhancing_image", 0, 1, f"Enhancing {image_path.name}...")

        try:
            enhance_image(
                image_path=image_path,
                output_path=image_path,
                softness=softness,
                quantize=image_settings.get("quantize", 8),
                tiled_vae=True,
            )
        except Exception as e:
            return PipelineResult(success=False, error=f"Enhancement failed: {e}")

        self._sync_file(image_path)
        self.on_image_ready(run_id, image_path.name)
        self.on_progress("enhancing_image", 1, 1, "Enhancement complete")

        return PipelineResult(
            success=True,
            run_id=run_id,
            image_count=1,
            data={"image_path": image_path.name},
        )

    def generate_all_images(
        self,
        run_id: str,
        images_per_prompt: int = 1,
        resume: bool = True,
    ) -> PipelineResult:
        """Generate all images for a gallery.

        Args:
            run_id: Run directory name
            images_per_prompt: Images per prompt
            resume: Skip existing images

        Returns:
            PipelineResult with operation results
        """
        output_dir = paths.prompts_dir / run_id

        if not output_dir.exists():
            return PipelineResult(success=False, error=f"Run directory not found: {run_id}")

        # Load metadata
        meta_files = list(output_dir.glob("*_metadata.json"))
        if not meta_files:
            return PipelineResult(success=False, error="No metadata file found")

        metadata = json.loads(meta_files[0].read_text())
        prefix = metadata.get("prefix", "image")
        image_settings = metadata.get("image_generation", {})

        # Load prompts
        prompt_files = sorted(output_dir.glob(f"{prefix}_*.txt"))
        prompt_files = [f for f in prompt_files if f.stem.count('_') == 1]

        if not prompt_files:
            return PipelineResult(success=False, error="No prompt files found")

        prompts = [f.read_text() for f in prompt_files]
        total_images = len(prompts) * images_per_prompt

        self.on_progress("generating_images", 0, total_images, "Starting image generation...")

        current_seed = image_settings.get("seed")
        generated_count = 0
        skipped_count = 0

        model = image_settings.get("model", "z-image-turbo")

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
                self.on_progress(
                    "generating_images",
                    generated_count,
                    total_images,
                    f"Generating {output_path.name}..."
                )

                try:
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
                    return PipelineResult(
                        success=False,
                        error=f"Image generation failed ({output_path.name}): {e}"
                    )

                self._sync_file(output_path)
                self.on_image_ready(run_id, output_path.name)

                if current_seed is not None:
                    current_seed += 1

        actual_generated = generated_count - skipped_count

        return PipelineResult(
            success=True,
            run_id=run_id,
            image_count=actual_generated,
            skipped_count=skipped_count,
        )

    def enhance_all_images(
        self,
        run_id: str,
        softness: float = 0.5,
    ) -> PipelineResult:
        """Enhance all images for a gallery.

        Args:
            run_id: Run directory name
            softness: Enhancement softness

        Returns:
            PipelineResult with operation results
        """
        output_dir = paths.prompts_dir / run_id

        if not output_dir.exists():
            return PipelineResult(success=False, error=f"Run directory not found: {run_id}")

        # Load metadata
        meta_files = list(output_dir.glob("*_metadata.json"))
        if not meta_files:
            return PipelineResult(success=False, error="No metadata file found")

        metadata = json.loads(meta_files[0].read_text())
        prefix = metadata.get("prefix", "image")
        image_settings = metadata.get("image_generation", {})

        # Find all images
        images = sorted(output_dir.glob(f"{prefix}_*_*.png"))
        if not images:
            return PipelineResult(success=False, error="No images found")

        # Create backup if not already done
        if not metadata.get("_enhancement_backup_created"):
            self.on_progress("backup", 0, 1, "Creating backup before enhancement...")
            try:
                backup_run(output_dir, paths.saved_dir, reason="pre_enhance")
                metadata["_enhancement_backup_created"] = datetime.now().isoformat()
                meta_files[0].write_text(json.dumps(metadata, indent=2))
            except Exception as e:
                logger.warning(f"Failed to create backup before enhancement: {e}")

        self.on_progress("enhancing_images", 0, len(images), "Starting enhancement...")

        for idx, image_path in enumerate(images, 1):
            self.on_progress(
                "enhancing_images",
                idx,
                len(images),
                f"Enhancing {image_path.name}..."
            )

            try:
                enhance_image(
                    image_path=image_path,
                    output_path=image_path,
                    softness=softness,
                    quantize=image_settings.get("quantize", 8),
                    tiled_vae=True,
                )
            except Exception as e:
                return PipelineResult(
                    success=False,
                    error=f"Enhancement failed ({image_path.name}): {e}"
                )

            self._sync_file(image_path)
            self.on_image_ready(run_id, image_path.name)

        return PipelineResult(
            success=True,
            run_id=run_id,
            image_count=len(images),
        )

    def _generate_images(
        self,
        output_dir: Path,
        run_id: str,
        prefix: str,
        prompts: list[str],
        images_per_prompt: int,
        model: str,
        steps: int | None,
        width: int,
        height: int,
        quantize: int,
        seed: int | None,
        tiled_vae: bool,
        enhance: bool,
        enhance_softness: float,
        enhance_after: bool,
        resume: bool,
        gallery_path: Path,
    ) -> PipelineResult:
        """Internal method to generate images with progress reporting."""
        total_images = len(prompts) * images_per_prompt
        self.on_progress("generating_images", 0, total_images, "Starting image generation...")

        current_seed = seed
        generated_count = 0
        skipped_count = 0
        images_to_enhance = []

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
                self.on_progress(
                    "generating_images",
                    generated_count,
                    total_images,
                    f"Generating {output_path.name}..."
                )

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
                        tiled_vae=tiled_vae,
                    )
                except Exception as e:
                    return PipelineResult(
                        success=False,
                        error=f"Image generation failed ({output_path.name}): {e}"
                    )

                self._sync_file(output_path)
                update_gallery(gallery_path, output_path, prompt_text, generated_count, total_images)
                self.on_image_ready(run_id, output_path.name)

                # Enhancement
                if enhance and not enhance_after:
                    self.on_progress(
                        "enhancing_image",
                        generated_count,
                        total_images,
                        f"Enhancing {output_path.name}..."
                    )
                    try:
                        enhance_image(
                            image_path=output_path,
                            output_path=output_path,
                            softness=enhance_softness,
                            seed=current_seed,
                            quantize=quantize,
                            tiled_vae=tiled_vae,
                        )
                    except Exception as e:
                        return PipelineResult(
                            success=False,
                            error=f"Enhancement failed ({output_path.name}): {e}"
                        )

                    self._sync_file(output_path)
                    self.on_image_ready(run_id, output_path.name)
                elif enhance and enhance_after:
                    images_to_enhance.append((output_path, current_seed))

                if current_seed is not None:
                    current_seed += 1

        # Batch enhancement
        if enhance and enhance_after and images_to_enhance:
            clear_model_cache()
            self.on_progress(
                "enhancing_images", 0, len(images_to_enhance),
                "Starting batch enhancement..."
            )

            for idx, (image_path, image_seed) in enumerate(images_to_enhance, 1):
                self.on_progress(
                    "enhancing_images",
                    idx,
                    len(images_to_enhance),
                    f"Enhancing {image_path.name}..."
                )
                try:
                    enhance_image(
                        image_path=image_path,
                        output_path=image_path,
                        softness=enhance_softness,
                        seed=image_seed,
                        quantize=quantize,
                        tiled_vae=tiled_vae,
                    )
                except Exception as e:
                    return PipelineResult(
                        success=False,
                        error=f"Enhancement failed ({image_path.name}): {e}"
                    )

                self._sync_file(image_path)
                self.on_image_ready(run_id, image_path.name)

        actual_generated = generated_count - skipped_count

        return PipelineResult(
            success=True,
            run_id=run_id,
            output_dir=output_dir,
            image_count=actual_generated,
            skipped_count=skipped_count,
        )

    @staticmethod
    def _sync_file(path: Path) -> None:
        """Ensure file is flushed to disk."""
        import os
        with open(path, 'r+b') as f:
            f.flush()
            os.fsync(f.fileno())
