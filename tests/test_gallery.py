"""Tests for interactive gallery generation."""

import json

from conftest import create_run_files
from gallery import generate_gallery_for_directory


class TestGalleryInteractive:
    """Tests for interactive gallery generation."""

    def test_gallery_interactive_mode(self, temp_dir, sample_grammar):
        """Test that interactive gallery includes editor and buttons."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 2,
            "user_prompt": "test prompt",
            "image_generation": {"images_per_prompt": 1},
        }
        create_run_files(run_dir, metadata=metadata, grammar=sample_grammar)

        # Generate interactive gallery
        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)

        assert gallery_path.exists()
        content = gallery_path.read_text()

        # Check for interactive elements
        assert "grammar-editor" in content
        assert "btn-save-grammar" in content
        assert "btn-regenerate" in content
        assert "btn-generate-all" in content
        assert "btn-enhance-all" in content
        assert "generateImage" in content
        assert "enhanceImage" in content

    def test_gallery_interactive_has_nav_and_archive(self, temp_dir):
        """Test that interactive gallery includes nav header and archive button."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 1,
            "user_prompt": "test prompt",
            "image_generation": {"images_per_prompt": 1},
        }
        create_run_files(run_dir, num_prompts=1, metadata=metadata)

        # Generate interactive gallery
        gallery_path = generate_gallery_for_directory(run_dir, interactive=True)
        content = gallery_path.read_text()

        # Check for nav header
        assert "nav-header" in content
        assert "Back to Index" in content
        assert "/index" in content

        # Check for archive button
        assert "btn-archive" in content
        assert "Save to Archive" in content

    def test_gallery_non_interactive_mode(self, temp_dir):
        """Test that non-interactive gallery doesn't include interactive elements."""
        run_dir = temp_dir
        metadata = {
            "prefix": "test",
            "count": 1,
            "user_prompt": "test",
            "image_generation": {"images_per_prompt": 1},
        }
        create_run_files(run_dir, num_prompts=1, metadata=metadata)

        # Generate non-interactive gallery
        gallery_path = generate_gallery_for_directory(run_dir, interactive=False)

        content = gallery_path.read_text()

        # Check that interactive elements are NOT present
        assert "grammar-editor" not in content
        assert "btn-generate-all" not in content
        assert "generateImage(" not in content
