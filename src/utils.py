"""Shared utility functions for the image-prompt-expander application."""

import json
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
