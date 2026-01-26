"""Shared utility functions for the image-prompt-expander application."""

import json
import shutil
from datetime import datetime
from pathlib import Path


def load_run_metadata(run_dir: Path) -> dict:
    """Load metadata from a run directory.

    Args:
        run_dir: Path to a run directory containing *_metadata.json

    Returns:
        Dictionary containing the metadata

    Raises:
        ValueError: If no metadata file found
        json.JSONDecodeError: If metadata file is invalid JSON
    """
    meta_files = list(run_dir.glob("*_metadata.json"))
    if not meta_files:
        raise ValueError(f"No metadata file found in {run_dir}")
    return json.loads(meta_files[0].read_text())


def get_prefix_from_metadata(run_dir: Path) -> str:
    """Get the prefix from a run directory's metadata.

    Args:
        run_dir: Path to a run directory

    Returns:
        The prefix string (defaults to "image" if not found)
    """
    try:
        metadata = load_run_metadata(run_dir)
        return metadata.get("prefix", "image")
    except (ValueError, json.JSONDecodeError):
        return "image"


def find_metadata_file(run_dir: Path) -> Path | None:
    """Find the metadata file in a run directory.

    Args:
        run_dir: Path to a run directory

    Returns:
        Path to the metadata file, or None if not found
    """
    meta_files = list(run_dir.glob("*_metadata.json"))
    return meta_files[0] if meta_files else None


def count_images_in_run(run_dir: Path, prefix: str | None = None) -> int:
    """Count the number of generated images in a run directory.

    Args:
        run_dir: Path to a run directory
        prefix: Optional prefix for image files (auto-detected if not provided)

    Returns:
        Number of image files found
    """
    if prefix is None:
        prefix = get_prefix_from_metadata(run_dir)
    return len(list(run_dir.glob(f"{prefix}_*_*.png")))


def get_prompts_from_run(run_dir: Path, prefix: str | None = None) -> list[str]:
    """Load all prompt texts from a run directory.

    Args:
        run_dir: Path to a run directory
        prefix: Optional prefix for prompt files (auto-detected if not provided)

    Returns:
        List of prompt strings in order
    """
    if prefix is None:
        prefix = get_prefix_from_metadata(run_dir)

    prompt_files = sorted(run_dir.glob(f"{prefix}_*.txt"))
    # Filter to only prompt files (prefix_N.txt), not metadata or other files
    prompt_files = [f for f in prompt_files if f.stem.count('_') == 1]

    return [f.read_text() for f in prompt_files]


def backup_run(run_dir: Path, saved_dir: Path, reason: str = "manual_archive") -> Path:
    """Create a backup of a run directory (images and metadata only).

    Only copies PNG images and the metadata file to minimize archive size.
    Prompts, logs, grammar, and gallery HTML are not included since they
    can be regenerated and archives are meant for preserving image results.

    Args:
        run_dir: Path to the run directory to backup
        saved_dir: Path to the saved/ directory
        reason: "pre_regenerate", "pre_enhance", or "manual_archive"

    Returns:
        Path to the created backup directory
    """
    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{run_dir.name}_{timestamp}"
    backup_dir = saved_dir / backup_name

    saved_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    prefix = get_prefix_from_metadata(run_dir)

    # Copy only PNG images (pattern: {prefix}_{prompt_idx}_{image_idx}.png)
    for png_file in run_dir.glob(f"{prefix}_*_*.png"):
        shutil.copy2(png_file, backup_dir / png_file.name)

    # Copy metadata file (required for archive to be browsable in index)
    meta_file = find_metadata_file(run_dir)
    if meta_file:
        dest_meta = backup_dir / meta_file.name
        shutil.copy2(meta_file, dest_meta)
        # Update with backup info
        metadata = json.loads(dest_meta.read_text())
        metadata["backup_info"] = {
            "is_backup": True,
            "source_run_id": run_dir.name,
            "backup_created_at": datetime.now().isoformat(),
            "backup_reason": reason,
        }
        dest_meta.write_text(json.dumps(metadata, indent=2))

    return backup_dir


def run_has_images(run_dir: Path) -> bool:
    """Check if a run directory contains any generated images."""
    prefix = get_prefix_from_metadata(run_dir)
    return len(list(run_dir.glob(f"{prefix}_*_*.png"))) > 0


def is_backup_run(run_dir: Path) -> bool:
    """Check if a run directory is a backup."""
    try:
        metadata = load_run_metadata(run_dir)
        return metadata.get("backup_info", {}).get("is_backup", False)
    except (ValueError, json.JSONDecodeError):
        return False
