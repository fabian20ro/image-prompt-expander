"""Tests for utility functions."""

import json

import pytest
from PIL import Image

from utils import (
    load_run_metadata,
    get_prefix_from_metadata,
    count_images_in_run,
    get_prompts_from_run,
    backup_run,
    is_backup_run,
    get_flat_archive_metadata,
    run_has_images,
    delete_run,
    format_run_timestamp,
)


class TestUtils:
    """Tests for utility functions."""

    def test_format_run_timestamp_valid(self):
        """Test formatting a valid timestamp."""
        result = format_run_timestamp("20240115_143022")
        assert result == "2024-01-15 14:30:22"

    def test_format_run_timestamp_invalid_length(self):
        """Test that invalid length returns original."""
        result = format_run_timestamp("20240115")
        assert result == "20240115"

    def test_format_run_timestamp_missing_separator(self):
        """Test that missing separator returns original."""
        result = format_run_timestamp("20240115-143022")
        assert result == "20240115-143022"

    def test_format_run_timestamp_empty(self):
        """Test that empty string returns empty."""
        result = format_run_timestamp("")
        assert result == ""

    def test_load_run_metadata(self, temp_dir):
        """Test loading metadata from a run directory."""
        (temp_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "count": 10,
            "user_prompt": "a dragon",
        }))

        metadata = load_run_metadata(temp_dir)
        assert metadata["prefix"] == "test"
        assert metadata["count"] == 10

    def test_load_run_metadata_not_found(self, temp_dir):
        """Test that ValueError is raised when no metadata found."""
        with pytest.raises(ValueError, match="No metadata file found"):
            load_run_metadata(temp_dir)

    def test_load_run_metadata_malformed_json(self, temp_dir):
        """Test loading metadata with malformed JSON."""
        (temp_dir / "test_metadata.json").write_text("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            load_run_metadata(temp_dir)

    def test_get_prefix_from_metadata(self, temp_dir):
        """Test getting prefix from metadata."""
        (temp_dir / "cat_metadata.json").write_text(json.dumps({
            "prefix": "cat",
        }))

        prefix = get_prefix_from_metadata(temp_dir)
        assert prefix == "cat"

    def test_get_prefix_from_metadata_default(self, temp_dir):
        """Test that default prefix is returned when metadata missing."""
        prefix = get_prefix_from_metadata(temp_dir)
        assert prefix == "image"

    def test_count_images_in_run(self, temp_dir):
        """Test counting images in a run directory."""
        (temp_dir / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))
        (temp_dir / "test_0_0.png").write_text("fake image")
        (temp_dir / "test_0_1.png").write_text("fake image")
        (temp_dir / "test_1_0.png").write_text("fake image")

        count = count_images_in_run(temp_dir)
        assert count == 3

    def test_get_prompts_from_run(self, temp_dir):
        """Test loading prompts from a run directory."""
        (temp_dir / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))
        (temp_dir / "test_0.txt").write_text("First prompt")
        (temp_dir / "test_1.txt").write_text("Second prompt")
        (temp_dir / "test_2.txt").write_text("Third prompt")

        prompts = get_prompts_from_run(temp_dir)
        assert len(prompts) == 3
        assert prompts[0] == "First prompt"
        assert prompts[2] == "Third prompt"

    def test_backup_run(self, temp_dir):
        """Test backing up a run directory to flat files with EXIF metadata."""
        run_dir = temp_dir / "prompts" / "20240101_120000_abc123"
        saved_dir = temp_dir / "saved"
        run_dir.mkdir(parents=True)

        # Create test files
        (run_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
            "model": "test-model",
        }))
        (run_dir / "test_0.txt").write_text("Prompt 0")

        # Create a real PNG file for Pillow to process
        img = Image.new('RGB', (10, 10), color='red')
        img.save(run_dir / "test_0_0.png")

        # Create backup
        saved_files = backup_run(run_dir, saved_dir, reason="pre_regenerate")

        # Check that files were saved
        assert len(saved_files) == 1
        assert saved_files[0].exists()

        # Check flat file naming pattern: prefix_timestamp_promptIdx_imgIdx.png
        filename = saved_files[0].name
        assert filename.startswith("test_")
        assert filename.endswith("_0_0.png")

        # Check that metadata is embedded in PNG
        metadata = get_flat_archive_metadata(saved_files[0])
        assert metadata.get("user_prompt") == "a dragon"
        assert metadata.get("model") == "test-model"
        assert metadata.get("backup_reason") == "pre_regenerate"
        assert metadata.get("prompt") == "Prompt 0"

        # Check is_backup_run still works for original run_dir
        assert is_backup_run(run_dir) is False

    def test_backup_run_not_found(self, temp_dir):
        """Test that backup fails for non-existent directory."""
        non_existent = temp_dir / "does_not_exist"
        saved_dir = temp_dir / "saved"

        with pytest.raises(ValueError, match="Run directory not found"):
            backup_run(non_existent, saved_dir)

    def test_backup_run_no_images(self, temp_dir):
        """Test backup with no images returns empty list."""
        run_dir = temp_dir / "prompts" / "20240101_120000_abc123"
        saved_dir = temp_dir / "saved"
        run_dir.mkdir(parents=True)

        # Create metadata but no images
        (run_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
        }))

        saved_files = backup_run(run_dir, saved_dir, reason="manual_archive")
        assert saved_files == []

    def test_run_has_images(self, temp_dir):
        """Test checking if a run has images."""
        (temp_dir / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))

        # No images yet
        assert run_has_images(temp_dir) is False

        # Add an image
        (temp_dir / "test_0_0.png").write_bytes(b"fake image")
        assert run_has_images(temp_dir) is True

    def test_delete_run(self, temp_dir):
        """Test deleting a run directory."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        # Create test files
        (run_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
        }))
        (run_dir / "test_0.txt").write_text("Prompt 0")
        (run_dir / "test_0_0.png").write_bytes(b"fake image")

        assert run_dir.exists()

        # Delete the run
        delete_run(run_dir, prompts_dir)

        assert not run_dir.exists()

    def test_delete_run_not_found(self, temp_dir):
        """Test that delete fails for non-existent directory."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()
        non_existent = prompts_dir / "does_not_exist"

        with pytest.raises(ValueError, match="Run directory not found"):
            delete_run(non_existent, prompts_dir)

    def test_delete_run_outside_prompts_dir(self, temp_dir):
        """Test that delete fails for directories outside prompts_dir."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()
        outside_dir = temp_dir / "outside"
        outside_dir.mkdir()
        (outside_dir / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))

        with pytest.raises(ValueError, match="not inside prompts directory"):
            delete_run(outside_dir, prompts_dir)

    def test_delete_run_archive_protected(self, temp_dir):
        """Test that delete fails for archived galleries."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        # Create backup metadata (marks this as an archive)
        (run_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "backup_info": {
                "is_backup": True,
                "source_run_id": "original",
                "backup_reason": "manual_archive",
            },
        }))

        with pytest.raises(ValueError, match="Cannot delete archived galleries"):
            delete_run(run_dir, prompts_dir)

    def test_delete_run_path_traversal(self, temp_dir):
        """Test that delete fails for path traversal attempts."""
        prompts_dir = temp_dir / "prompts"
        prompts_dir.mkdir()

        # Attempt path traversal
        traversal_path = prompts_dir / ".." / "outside"
        traversal_path.mkdir(parents=True)
        (traversal_path / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))

        with pytest.raises(ValueError, match="not inside prompts directory"):
            delete_run(traversal_path, prompts_dir)

    def test_get_flat_archive_metadata_corrupted_png(self, temp_dir):
        """Test get_flat_archive_metadata with corrupted PNG returns empty dict."""
        corrupted_file = temp_dir / "corrupted.png"
        corrupted_file.write_bytes(b"not a valid png file")

        metadata = get_flat_archive_metadata(corrupted_file)
        assert metadata == {}

    def test_get_flat_archive_metadata_missing_file(self, temp_dir):
        """Test get_flat_archive_metadata with missing file returns empty dict."""
        missing_file = temp_dir / "missing.png"

        metadata = get_flat_archive_metadata(missing_file)
        assert metadata == {}
