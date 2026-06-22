"""Shared test fixtures for all test modules."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Match the flat-module runtime used by `uv run python src/cli.py`.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def run_dir(temp_dir):
    """Create a mock run directory with basic structure."""
    run_dir = temp_dir / "prompts" / "20240101_120000_abc123"
    run_dir.mkdir(parents=True)
    return run_dir


@pytest.fixture
def saved_dir(temp_dir):
    """Create a saved/archive directory."""
    saved_dir = temp_dir / "saved"
    saved_dir.mkdir(parents=True)
    return saved_dir


@pytest.fixture
def queue_path(temp_dir):
    """Create a queue file path for testing."""
    return temp_dir / "queue.json"


@pytest.fixture
def sample_metadata():
    """Sample metadata dictionary for testing."""
    return {
        "prefix": "test",
        "count": 10,
        "user_prompt": "a dragon flying over mountains",
        "model": "ernie-image-turbo",
        "image_generation": {"images_per_prompt": 1},
    }


@pytest.fixture
def sample_grammar():
    """Sample Tracery grammar for testing."""
    return {
        "origin": ["#subject# in #setting#"],
        "subject": ["a cat", "a dog"],
        "setting": ["a garden", "a forest"],
    }


def create_run_files(run_dir: Path, prefix: str = "test", num_prompts: int = 2,
                      metadata: dict = None, grammar: dict = None, create_images: bool = False):
    """Helper to create run directory with files.

    Args:
        run_dir: Directory to create files in
        prefix: File prefix
        num_prompts: Number of prompt files to create
        metadata: Metadata dict (defaults to basic metadata)
        grammar: Grammar dict (defaults to simple grammar)
        create_images: Whether to create fake image files
    """
    if metadata is None:
        metadata = {
            "prefix": prefix,
            "count": num_prompts,
            "user_prompt": "test prompt",
            "image_generation": {"images_per_prompt": 1},
        }

    if grammar is None:
        grammar = {"origin": ["test"]}

    # Ensure directory exists
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata
    (run_dir / f"{prefix}.metaprompt.json").write_text(json.dumps(metadata))

    # Create grammar
    (run_dir / f"{prefix}_grammar.json").write_text(json.dumps(grammar))

    # Create prompts
    for i in range(num_prompts):
        (run_dir / f"{prefix}_{i}.txt").write_text(f"Prompt {i}")

    # Create images if requested
    if create_images:
        for i in range(num_prompts):
            (run_dir / f"{prefix}_{i}_0.png").write_bytes(b"fake image")

    return run_dir


@pytest.fixture
def full_run_dir(run_dir):
    """Create a run directory with all required files."""
    return create_run_files(run_dir)


def test_create_run_files_integrity(temp_dir):
    """Verify create_run_files creates a valid directory structure."""
    num_prompts = 3
    prefix = "test_run"
    run_dir = create_run_files(temp_dir / "test_run", prefix=prefix, num_prompts=num_prompts)
    assert (run_dir / f"{prefix}.metaprompt.json").exists()
    assert (run_dir / f"{prefix}_grammar.json").exists()
    for i in range(num_prompts):
        assert (run_dir / f"{prefix}_{i}.txt").exists()
