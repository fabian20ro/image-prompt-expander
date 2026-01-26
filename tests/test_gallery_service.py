"""Tests for GalleryService."""

import json

import pytest

from services.gallery_service import (
    GalleryService,
    GalleryNotFoundError,
    MetadataNotFoundError,
)
from conftest import create_run_files


class TestGalleryService:
    """Tests for GalleryService."""

    def test_get_run_directory_success(self, temp_dir):
        """Test getting a valid run directory."""
        prompts_dir = temp_dir / "prompts"
        saved_dir = temp_dir / "saved"
        prompts_dir.mkdir()
        run_dir = prompts_dir / "20240101_120000_abc123"
        run_dir.mkdir()

        service = GalleryService(prompts_dir, saved_dir)
        result = service.get_run_directory("20240101_120000_abc123")
        assert result == run_dir

    def test_get_run_directory_not_found(self, temp_dir):
        """Test getting a non-existent run directory."""
        prompts_dir = temp_dir / "prompts"
        saved_dir = temp_dir / "saved"
        prompts_dir.mkdir()

        service = GalleryService(prompts_dir, saved_dir)
        with pytest.raises(GalleryNotFoundError, match="Gallery not found"):
            service.get_run_directory("nonexistent")

    def test_get_run_directory_archive(self, temp_dir):
        """Test getting an archive directory."""
        prompts_dir = temp_dir / "prompts"
        saved_dir = temp_dir / "saved"
        saved_dir.mkdir()
        archive_dir = saved_dir / "archive_20240101"
        archive_dir.mkdir()

        service = GalleryService(prompts_dir, saved_dir)
        result = service.get_run_directory("archive_20240101", is_archive=True)
        assert result == archive_dir

    def test_get_run_directory_archive_not_found(self, temp_dir):
        """Test getting a non-existent archive directory."""
        prompts_dir = temp_dir / "prompts"
        saved_dir = temp_dir / "saved"
        saved_dir.mkdir()

        service = GalleryService(prompts_dir, saved_dir)
        with pytest.raises(GalleryNotFoundError, match="Archive not found"):
            service.get_run_directory("nonexistent", is_archive=True)

    def test_load_metadata_success(self, temp_dir):
        """Test loading metadata."""
        (temp_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
        }))

        service = GalleryService(temp_dir, temp_dir)
        metadata = service.load_metadata(temp_dir)

        assert metadata["prefix"] == "test"
        assert metadata["user_prompt"] == "a dragon"

    def test_load_metadata_not_found(self, temp_dir):
        """Test loading metadata when not found."""
        service = GalleryService(temp_dir, temp_dir)
        with pytest.raises(MetadataNotFoundError, match="No metadata file found"):
            service.load_metadata(temp_dir)

    def test_load_metadata_malformed_json(self, temp_dir):
        """Test loading metadata with malformed JSON."""
        (temp_dir / "test_metadata.json").write_text("not valid json")

        service = GalleryService(temp_dir, temp_dir)
        with pytest.raises(json.JSONDecodeError):
            service.load_metadata(temp_dir)

    def test_get_prefix(self, temp_dir):
        """Test getting prefix from metadata."""
        (temp_dir / "cat_metadata.json").write_text(json.dumps({"prefix": "cat"}))

        service = GalleryService(temp_dir, temp_dir)
        assert service.get_prefix(temp_dir) == "cat"

    def test_get_prefix_default(self, temp_dir):
        """Test getting default prefix when no metadata."""
        service = GalleryService(temp_dir, temp_dir)
        assert service.get_prefix(temp_dir) == "image"

    def test_load_grammar_success(self, temp_dir):
        """Test loading grammar."""
        grammar = {"origin": ["#subject#"], "subject": ["cat"]}
        (temp_dir / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))
        (temp_dir / "test_grammar.json").write_text(json.dumps(grammar))

        service = GalleryService(temp_dir, temp_dir)
        result = service.load_grammar(temp_dir, "test")

        assert '"origin"' in result
        assert '"subject"' in result

    def test_load_grammar_not_found(self, temp_dir):
        """Test loading grammar when file doesn't exist."""
        service = GalleryService(temp_dir, temp_dir)
        result = service.load_grammar(temp_dir, "test")
        assert result is None

    def test_load_prompts(self, temp_dir):
        """Test loading prompts."""
        (temp_dir / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))
        (temp_dir / "test_0.txt").write_text("First prompt")
        (temp_dir / "test_1.txt").write_text("Second prompt")
        (temp_dir / "test_2.txt").write_text("Third prompt")
        # Should NOT include these (more than 1 underscore):
        (temp_dir / "test_0_0.png").write_bytes(b"fake")
        (temp_dir / "test_not_a_prompt.txt").write_text("other")

        service = GalleryService(temp_dir, temp_dir)
        prompts = service.load_prompts(temp_dir)

        assert len(prompts) == 3
        assert prompts[0] == "First prompt"
        assert prompts[1] == "Second prompt"
        assert prompts[2] == "Third prompt"

    def test_load_prompts_with_missing_files(self, temp_dir):
        """Test loading prompts when directory is empty."""
        (temp_dir / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))

        service = GalleryService(temp_dir, temp_dir)
        prompts = service.load_prompts(temp_dir)

        assert prompts == []

    def test_validate_file_access_valid(self, temp_dir):
        """Test validating file access for valid path."""
        file_path = temp_dir / "subdir" / "file.txt"

        service = GalleryService(temp_dir, temp_dir)
        result = service.validate_file_access(file_path, temp_dir)

        assert result is True

    def test_validate_file_access_invalid_path_traversal(self, temp_dir):
        """Test validating file access blocks path traversal."""
        base_dir = temp_dir / "base"
        base_dir.mkdir()
        outside_file = temp_dir / "outside" / "file.txt"

        service = GalleryService(temp_dir, temp_dir)
        with pytest.raises(ValueError, match="Access denied"):
            service.validate_file_access(outside_file, base_dir)

    def test_is_backup_run_false(self, temp_dir):
        """Test is_backup_run returns False for normal runs."""
        (temp_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "user_prompt": "a dragon",
        }))

        service = GalleryService(temp_dir, temp_dir)
        assert service.is_backup_run(temp_dir) is False

    def test_is_backup_run_true(self, temp_dir):
        """Test is_backup_run returns True for backup runs."""
        (temp_dir / "test_metadata.json").write_text(json.dumps({
            "prefix": "test",
            "backup_info": {
                "is_backup": True,
                "source_run_id": "original",
            },
        }))

        service = GalleryService(temp_dir, temp_dir)
        assert service.is_backup_run(temp_dir) is True

    def test_is_backup_run_no_metadata(self, temp_dir):
        """Test is_backup_run returns False when no metadata."""
        service = GalleryService(temp_dir, temp_dir)
        assert service.is_backup_run(temp_dir) is False

    def test_count_images(self, temp_dir):
        """Test counting images."""
        (temp_dir / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))
        (temp_dir / "test_0_0.png").write_bytes(b"fake")
        (temp_dir / "test_0_1.png").write_bytes(b"fake")
        (temp_dir / "test_1_0.png").write_bytes(b"fake")
        # Should NOT count these:
        (temp_dir / "test_0.txt").write_text("prompt")
        (temp_dir / "other.png").write_bytes(b"fake")

        service = GalleryService(temp_dir, temp_dir)
        count = service.count_images(temp_dir)

        assert count == 3

    def test_list_images(self, temp_dir):
        """Test listing images."""
        (temp_dir / "test_metadata.json").write_text(json.dumps({"prefix": "test"}))
        (temp_dir / "test_0_0.png").write_bytes(b"fake")
        (temp_dir / "test_1_0.png").write_bytes(b"fake")
        (temp_dir / "test_0_1.png").write_bytes(b"fake")

        service = GalleryService(temp_dir, temp_dir)
        images = service.list_images(temp_dir)

        assert len(images) == 3
        # Check sorted order
        assert images[0].name == "test_0_0.png"
        assert images[1].name == "test_0_1.png"
        assert images[2].name == "test_1_0.png"
